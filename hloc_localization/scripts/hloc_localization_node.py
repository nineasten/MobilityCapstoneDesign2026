#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


@dataclass(frozen=True)
class PendingFrame:
    msg: Image
    camera_info: Optional[CameraInfo]


@dataclass(frozen=True)
class CameraIntrinsics:
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class LocalizationResult:
    query_name: str
    qvec_wxyz: np.ndarray
    tvec_cam_from_world: np.ndarray


class HlocLocalizationNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.pending_frame: Optional[PendingFrame] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.last_process_time = rospy.Time(0)
        self.worker_error_logged = False

        self.image_topic = str(rospy.get_param("~image_topic", "/arducam_imx219/image_raw"))
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", ""))
        self.pose_topic = str(rospy.get_param("~pose_topic", "/planner/current_pose"))
        self.status_topic = str(rospy.get_param("~status_topic", "/hloc_localization/status"))
        self.frame_id = str(rospy.get_param("~frame_id", "map"))

        self.reference_dir = self.resolve_reference_dir(
            rospy.get_param("~reference_dir", "")
        )
        runtime_param = str(rospy.get_param("~runtime_dir", "")).strip()
        self.runtime_dir = (
            Path(runtime_param).expanduser()
            if runtime_param
            else self.reference_dir / "runtime_queries" / "ros_live"
        )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.min_interval = float(rospy.get_param("~min_interval", 2.0))
        self.num_matched = int(rospy.get_param("~num_matched", 20))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95))
        self.covisibility_clustering = self.get_bool_param(
            "~covisibility_clustering", False
        )

        self.fx_param = float(rospy.get_param("~fx", 0.0))
        self.fy_param = float(rospy.get_param("~fy", 0.0))
        self.cx_param = float(rospy.get_param("~cx", 0.0))
        self.cy_param = float(rospy.get_param("~cy", 0.0))

        self.reference_global_descriptors = (
            self.reference_dir / "features" / "global-feats-netvlad.h5"
        )
        self.reference_local_features = (
            self.reference_dir / "features" / "feats-superpoint-n4096-rmax1600.h5"
        )
        self.sfm_dir = self.reference_dir / "sfm"

        self.validate_reference()
        self.load_hloc_modules()
        self.load_calibration()

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10)

        if self.camera_info_topic:
            self.camera_info_sub = rospy.Subscriber(
                self.camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
                queue_size=1,
            )
        else:
            self.camera_info_sub = None

        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        self.worker = threading.Thread(
            target=self.worker_loop,
            name="hloc_localization_worker",
            daemon=True,
        )
        self.worker.start()

        rospy.loginfo(
            "HLoc localization node ready: image=%s pose=%s reference=%s runtime=%s",
            self.image_topic,
            self.pose_topic,
            self.reference_dir,
            self.runtime_dir,
        )
        if not self.camera_info_topic:
            rospy.logwarn(
                "No camera_info_topic configured; using params or scaled default Arducam intrinsics."
            )

    @staticmethod
    def get_bool_param(name: str, default: bool) -> bool:
        value = rospy.get_param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def resolve_reference_dir(param_value: object) -> Path:
        if isinstance(param_value, str) and param_value.strip():
            return Path(param_value).expanduser().resolve()

        package_path = Path(rospkg.RosPack().get_path("hloc_localization"))
        workspace_src = package_path.parents[1]
        return (workspace_src / "hloc_reference").resolve()

    def validate_reference(self) -> None:
        required = [
            self.reference_global_descriptors,
            self.reference_local_features,
            self.sfm_dir / "cameras.bin",
            self.sfm_dir / "images.bin",
            self.sfm_dir / "points3D.bin",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "HLoc reference is incomplete. Missing: " + ", ".join(missing)
            )

    def load_hloc_modules(self) -> None:
        try:
            from hloc import extract_features, localize_sfm, match_features
            from hloc import pairs_from_retrieval
        except ImportError as error:
            raise ImportError(
                "Missing HLoc dependencies. Install with: "
                "python3 -m pip install h5py pycolmap "
                "git+https://github.com/cvg/Hierarchical-Localization.git"
            ) from error

        self.extract_features = extract_features
        self.match_features = match_features
        self.pairs_from_retrieval = pairs_from_retrieval
        self.localize_sfm = localize_sfm

    def load_calibration(self) -> None:
        calibration_param = str(rospy.get_param("~calibration_file", "")).strip()
        if calibration_param:
            calibration_path = Path(calibration_param).expanduser()
        else:
            calibration_path = self.reference_dir.parent / "calibration.json"

        if not calibration_path.exists():
            rospy.logwarn(
                "Calibration file not found (%s); publishing raw COLMAP world pose.",
                calibration_path,
            )
            self.calibration_scale = 1.0
            self.calibration_rotation = np.eye(3, dtype=np.float64)
            self.calibration_translation = np.zeros(3, dtype=np.float64)
            return

        with calibration_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.calibration_scale = float(data.get("scale", 1.0))
        self.calibration_rotation = np.asarray(
            data.get("rotation_matrix", np.eye(3)),
            dtype=np.float64,
        ).reshape(3, 3)
        self.calibration_translation = np.asarray(
            data.get("translation", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        rospy.loginfo("Loaded HLoc map calibration from %s", calibration_path)

    def camera_info_callback(self, msg: CameraInfo) -> None:
        with self.lock:
            self.latest_camera_info = msg

    def image_callback(self, msg: Image) -> None:
        now = rospy.Time.now()
        if (
            self.min_interval > 0.0
            and self.last_process_time != rospy.Time(0)
            and (now - self.last_process_time).to_sec() < self.min_interval
        ):
            return

        with self.lock:
            self.pending_frame = PendingFrame(msg=msg, camera_info=self.latest_camera_info)
            self.frame_event.set()

    def worker_loop(self) -> None:
        while not rospy.is_shutdown():
            if not self.frame_event.wait(0.2):
                continue

            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
                self.frame_event.clear()

            if frame is None:
                continue

            self.last_process_time = rospy.Time.now()
            try:
                self.process_frame(frame)
                self.worker_error_logged = False
            except Exception as error:  # noqa: BLE001 - keep node alive after bad frames.
                self.publish_status(f"failed: {error}")
                if not self.worker_error_logged:
                    rospy.logerr("HLoc localization failed: %s", error)
                    rospy.logdebug(traceback.format_exc())
                    self.worker_error_logged = True

    def process_frame(self, frame: PendingFrame) -> None:
        image_bgr = self.bridge.imgmsg_to_cv2(frame.msg, desired_encoding="bgr8")
        if image_bgr is None or image_bgr.size == 0:
            raise CvBridgeError("empty image")

        query_name = "query.jpg"
        query_path = self.runtime_dir / query_name
        if not cv2.imwrite(
            str(query_path),
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        ):
            raise RuntimeError(f"failed to write query image: {query_path}")

        intrinsics = self.resolve_intrinsics(image_bgr, frame.camera_info)
        queries_path = self.runtime_dir / "queries_with_intrinsics.txt"
        queries_path.write_text(
            self.format_query_intrinsics(query_name, intrinsics),
            encoding="utf-8",
        )

        pairs_path = self.runtime_dir / "pairs.txt"
        query_global_features = self.runtime_dir / "global-feats-netvlad.h5"
        query_local_features = self.runtime_dir / "feats-superpoint-n4096-rmax1600.h5"
        matches_path = self.runtime_dir / "matches.h5"
        localization_path = self.runtime_dir / "localization.txt"

        self.extract_features.main(
            self.extract_features.confs["netvlad"],
            self.runtime_dir,
            image_list=[query_name],
            feature_path=query_global_features,
            overwrite=True,
        )
        self.pairs_from_retrieval.main(
            query_global_features,
            pairs_path,
            num_matched=self.num_matched,
            query_list=[query_name],
            db_descriptors=self.reference_global_descriptors,
        )
        self.extract_features.main(
            self.extract_features.confs["superpoint_max"],
            self.runtime_dir,
            image_list=[query_name],
            feature_path=query_local_features,
            overwrite=True,
        )
        self.match_features.main(
            self.match_features.confs["superglue"],
            pairs_path,
            features=query_local_features,
            features_ref=self.reference_local_features,
            matches=matches_path,
            overwrite=True,
        )
        self.localize_sfm.main(
            self.sfm_dir,
            queries_path,
            pairs_path,
            query_local_features,
            matches_path,
            localization_path,
            covisibility_clustering=self.covisibility_clustering,
        )

        result = self.read_localization(localization_path, query_name)
        if result is None:
            self.publish_status("not_localized")
            rospy.logwarn_throttle(5.0, "HLoc could not localize the current frame.")
            return

        pose = self.result_to_pose(result, frame.msg.header.stamp)
        self.pose_pub.publish(pose)
        self.publish_status(
            "localized %.3f %.3f %.3f"
            % (
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            )
        )
        rospy.loginfo_throttle(
            2.0,
            "HLoc pose published: xyz=[%.3f, %.3f, %.3f]",
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )

    def resolve_intrinsics(
        self,
        image_bgr: np.ndarray,
        camera_info: Optional[CameraInfo],
    ) -> CameraIntrinsics:
        height, width = image_bgr.shape[:2]

        if camera_info is not None and len(camera_info.K) >= 6:
            fx = float(camera_info.K[0])
            fy = float(camera_info.K[4])
            cx = float(camera_info.K[2])
            cy = float(camera_info.K[5])
            if fx > 0.0 and fy > 0.0:
                return CameraIntrinsics("PINHOLE", width, height, (fx, fy, cx, cy))

        if self.fx_param > 0.0 and self.fy_param > 0.0:
            cx = self.cx_param if self.cx_param > 0.0 else width * 0.5
            cy = self.cy_param if self.cy_param > 0.0 else height * 0.5
            return CameraIntrinsics(
                "PINHOLE",
                width,
                height,
                (self.fx_param, self.fy_param, cx, cy),
            )

        # Current map package uses the Arducam B0495 1920x1200 FULL_OPENCV
        # calibration. Scale focal center terms if a resized stream is used.
        scale_x = width / 1920.0
        scale_y = height / 1200.0
        return CameraIntrinsics(
            "FULL_OPENCV",
            width,
            height,
            (
                1216.44616313 * scale_x,
                1214.18811858 * scale_y,
                956.646819947 * scale_x,
                583.900125334 * scale_y,
                -0.0819699141448,
                -0.03313321259,
                -0.00244342779542,
                -0.000488657078362,
                0.0165047957227,
                0.0,
                0.0,
                0.0,
            ),
        )

    @staticmethod
    def format_query_intrinsics(
        query_name: str,
        intrinsics: CameraIntrinsics,
    ) -> str:
        params = " ".join(f"{value:.9g}" for value in intrinsics.params)
        return (
            f"{query_name} {intrinsics.model} "
            f"{intrinsics.width} {intrinsics.height} {params}\n"
        )

    @staticmethod
    def read_localization(
        localization_path: Path,
        query_name: str,
    ) -> Optional[LocalizationResult]:
        if not localization_path.exists():
            return None

        for raw_line in localization_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) != 8 or parts[0] != query_name:
                continue
            values = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
            return LocalizationResult(
                query_name=parts[0],
                qvec_wxyz=values[:4],
                tvec_cam_from_world=values[4:],
            )
        return None

    def result_to_pose(self, result: LocalizationResult, stamp: rospy.Time) -> PoseStamped:
        rotation_world_to_cam = self.qvec_to_rotmat(result.qvec_wxyz)
        rotation_cam_to_world = rotation_world_to_cam.T
        center_world = -rotation_cam_to_world @ result.tvec_cam_from_world

        center_map = (
            self.calibration_scale
            * (self.calibration_rotation @ center_world)
            + self.calibration_translation
        )
        rotation_cam_to_map = self.calibration_rotation @ rotation_cam_to_world
        qx, qy, qz, qw = self.rotmat_to_quat_xyzw(rotation_cam_to_map)

        pose = PoseStamped()
        pose.header.stamp = stamp if stamp and not stamp.is_zero() else rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(center_map[0])
        pose.pose.position.y = float(center_map[1])
        pose.pose.position.z = float(center_map[2])
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        return pose

    @staticmethod
    def qvec_to_rotmat(qvec_wxyz: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = qvec_wxyz / np.linalg.norm(qvec_wxyz)
        return np.asarray(
            [
                [
                    1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                    2.0 * qx * qy - 2.0 * qz * qw,
                    2.0 * qx * qz + 2.0 * qy * qw,
                ],
                [
                    2.0 * qx * qy + 2.0 * qz * qw,
                    1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                    2.0 * qy * qz - 2.0 * qx * qw,
                ],
                [
                    2.0 * qx * qz - 2.0 * qy * qw,
                    2.0 * qy * qz + 2.0 * qx * qw,
                    1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def rotmat_to_quat_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale

        quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        quat /= np.linalg.norm(quat)
        return tuple(float(value) for value in quat)

    def publish_status(self, message: str) -> None:
        self.status_pub.publish(String(data=message))

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("hloc_localization")
    node = HlocLocalizationNode()
    node.spin()


if __name__ == "__main__":
    main()
