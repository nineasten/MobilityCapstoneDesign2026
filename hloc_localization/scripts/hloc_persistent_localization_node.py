#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import h5py
import numpy as np
import pycolmap
import rospkg
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


@dataclass(frozen=True)
class PendingFrame:
    msg: Image
    camera_info: Optional[CameraInfo]


@dataclass(frozen=True)
class CalibrationMetadata:
    scale: float
    rotation_matrix: np.ndarray
    translation: np.ndarray

    @classmethod
    def identity(cls) -> "CalibrationMetadata":
        return cls(
            scale=1.0,
            rotation_matrix=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "CalibrationMetadata":
        if not path.exists():
            rospy.logwarn(
                "Calibration file not found (%s); publishing raw COLMAP world pose.",
                path,
            )
            return cls.identity()

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        metadata = cls(
            scale=float(payload.get("scale", 1.0)),
            rotation_matrix=np.asarray(
                payload.get("rotation_matrix", np.eye(3)),
                dtype=np.float64,
            ).reshape(3, 3),
            translation=np.asarray(
                payload.get("translation", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(3),
        )
        rospy.loginfo("Loaded HLoc map calibration from %s", path)
        return metadata

    def apply_camera_pose(
        self,
        qvec_wxyz: Iterable[float],
        tvec_cam_from_world: Iterable[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation_world_to_cam = qvec_wxyz_to_rotmat(qvec_wxyz)
        rotation_cam_to_world = rotation_world_to_cam.T
        tvec = np.asarray(list(tvec_cam_from_world), dtype=np.float64).reshape(3)
        center_world = -(rotation_cam_to_world @ tvec)

        center_map = self.scale * (self.rotation_matrix @ center_world) + self.translation
        rotation_cam_to_map = self.rotation_matrix @ rotation_cam_to_world
        return center_map, rotmat_to_quat_xyzw(rotation_cam_to_map)


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def colmap_camera_line(
    image_name: str,
    model: str,
    width: int,
    height: int,
    params: Iterable[float],
) -> str:
    values = " ".join(format_float(value) for value in params)
    return f"{image_name} {model} {width} {height} {values}"


def has_nonzero_distortion(values: Iterable[float], eps: float = 1e-12) -> bool:
    return any(abs(float(value)) > eps for value in values)


def qvec_wxyz_to_rotmat(qvec_wxyz: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(list(qvec_wxyz), dtype=np.float64)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0:
        raise ValueError("invalid zero-length COLMAP quaternion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
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


def rotmat_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
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
    return quat


def load_hloc_modules() -> dict:
    from hloc import extract_features, extractors, localize_sfm, match_features, matchers
    from hloc.match_features import FeaturePairsDataset
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.parsers import names_to_pair

    return {
        "extract_features": extract_features,
        "extractors": extractors,
        "localize_sfm": localize_sfm,
        "match_features": match_features,
        "matchers": matchers,
        "FeaturePairsDataset": FeaturePairsDataset,
        "dynamic_load": dynamic_load,
        "names_to_pair": names_to_pair,
    }


def resolve_manifest_paths(manifest: dict, reference_dir: Path) -> dict:
    defaults = {
        "sfm_dir": Path("sfm"),
        "global_descriptors": Path("features/global-feats-netvlad.h5"),
        "local_features": Path("features/feats-superpoint-n4096-rmax1600.h5"),
    }
    resolved = dict(manifest)
    for key, fallback in defaults.items():
        raw = Path(str(resolved.get(key, fallback)))
        candidate = raw if raw.is_absolute() else reference_dir / raw
        if not candidate.exists():
            candidate = reference_dir / fallback
        resolved[key] = str(candidate.resolve())

    resolved.setdefault("retrieval_conf", "netvlad")
    resolved.setdefault("local_feature_conf", "superpoint_max")
    return resolved


def symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_feature(
    path: Path,
    name: str,
    pred: dict,
    uncertainty: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "a", libver="latest") as fd:
        if name in fd:
            del fd[name]
        group = fd.create_group(name)
        for key, value in pred.items():
            group.create_dataset(key, data=value)
        if uncertainty is not None and "keypoints" in group:
            group["keypoints"].attrs["uncertainty"] = uncertainty


class PersistentHlocLocalizer:
    def __init__(
        self,
        *,
        reference_dir: Path,
        calibration: CalibrationMetadata,
        runtime_root: Path,
        top_k: int,
        max_keypoints: int,
        resize_max: int,
        matcher_name: str,
        ransac_thresh: int,
        min_inliers: int,
        confidence_scale_inliers: float,
        device: str,
        fp16: bool = True,
    ) -> None:
        self.reference_dir = reference_dir.resolve()
        self.calibration = calibration
        self.runtime_root = runtime_root.resolve()
        self.top_k = top_k
        self.max_keypoints = max_keypoints
        self.resize_max = resize_max
        self.matcher_name = matcher_name
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.confidence_scale_inliers = confidence_scale_inliers
        self.device = self.resolve_device(device)
        self.use_fp16 = bool(fp16) and self.device == "cuda"

        self.runtime_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.reference_dir / "reference_manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = resolve_manifest_paths(manifest, self.reference_dir)
        self.validate_reference()

        rospy.loginfo(
            "Loading persistent HLoc models: device=%s top_k=%d max_keypoints=%d "
            "resize_max=%d matcher=%s",
            self.device,
            self.top_k,
            self.max_keypoints,
            self.resize_max,
            self.matcher_name,
        )
        started = time.perf_counter()
        self.modules = load_hloc_modules()
        try:
            self._load_models()
        except RuntimeError as error:
            if self.device == "cuda" and "CUDA" in str(error):
                rospy.logwarn(
                    "Could not load HLoc models on CUDA (%s); retrying on CPU.",
                    error,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.device = "cpu"
                self._load_models()
            else:
                raise
        rospy.loginfo("Persistent HLoc worker ready in %.3fs", time.perf_counter() - started)

    @staticmethod
    def resolve_device(requested: str) -> str:
        requested = str(requested or "auto").strip().lower()
        if requested in {"", "auto"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            rospy.logwarn("CUDA was requested, but torch.cuda is unavailable; using CPU.")
            return "cpu"
        if requested not in {"cuda", "cpu"}:
            rospy.logwarn("Unsupported HLoc device '%s'; using auto selection.", requested)
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested

    def validate_reference(self) -> None:
        required = [
            Path(self.manifest["global_descriptors"]),
            Path(self.manifest["local_features"]),
            Path(self.manifest["sfm_dir"]) / "cameras.bin",
            Path(self.manifest["sfm_dir"]) / "images.bin",
            Path(self.manifest["sfm_dir"]) / "points3D.bin",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "HLoc reference is incomplete. Missing: " + ", ".join(missing)
            )

    def _load_model(self, root, conf: dict, use_fp16: bool = False):
        model_cls = self.modules["dynamic_load"](root, conf["name"])
        model = model_cls(conf).eval().to(self.device)
        if use_fp16:
            model = model.half()
        return model

    def _load_models(self) -> None:
        extract_features = self.modules["extract_features"]
        match_features = self.modules["match_features"]
        self.retrieval_conf = extract_features.confs[self.manifest["retrieval_conf"]]
        self.local_conf = {
            "output": f"feats-superpoint-n{self.max_keypoints}-rmax{self.resize_max}",
            "model": {
                "name": "superpoint",
                "nms_radius": 3,
                "max_keypoints": self.max_keypoints,
            },
            "preprocessing": {
                "grayscale": True,
                "resize_max": self.resize_max,
                "resize_force": True,
            },
        }
        self.matcher_conf = match_features.confs[self.matcher_name]

        self.netvlad = self._load_model(
            self.modules["extractors"], self.retrieval_conf["model"], self.use_fp16
        )
        self.superpoint = self._load_model(
            self.modules["extractors"], self.local_conf["model"], self.use_fp16
        )
        # Matcher weights stay FP32 (FeaturePairsDataset always yields FP32
        # tensors); FP16 speedup is applied via autocast in match_pairs(),
        # which LightGlue itself handles internally (casts descriptors to
        # half for attention while keeping norms/softmax in FP32).
        self.matcher = self._load_model(
            self.modules["matchers"], self.matcher_conf["model"]
        )

        self.reconstruction = pycolmap.Reconstruction(Path(self.manifest["sfm_dir"]))
        self.db_names = [image.name for _, image in self.reconstruction.images.items()]
        self.db_desc = self._load_db_global_descriptors(
            self.db_names, Path(self.manifest["global_descriptors"])
        ).to(self.device)
        self.ref_local_features = Path(self.manifest["local_features"])

    def _load_db_global_descriptors(self, names: Iterable[str], path: Path) -> torch.Tensor:
        descriptors = []
        with h5py.File(str(path), "r", libver="latest") as fd:
            for name in names:
                descriptors.append(fd[name]["global_descriptor"].__array__())
        return torch.from_numpy(np.stack(descriptors, 0)).float()

    @torch.no_grad()
    def extract_one(
        self,
        model,
        conf: dict,
        image_dir: Path,
        image_name: str,
        output_path: Path,
    ) -> Path:
        dataset = self.modules["extract_features"].ImageDataset(
            image_dir,
            conf["preprocessing"],
            paths=[image_name],
        )
        data = dataset[0]
        image = torch.from_numpy(data["image"][None]).to(self.device)
        if self.use_fp16:
            image = image.half()
        original_size = data["original_size"]
        pred = model({"image": image})
        pred = {key: value[0].detach().float().cpu().numpy() for key, value in pred.items()}
        pred["image_size"] = original_size

        uncertainty = None
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        for key in list(pred):
            if pred[key].dtype == np.float32:
                pred[key] = pred[key].astype(np.float16)
        write_feature(output_path, image_name, pred, uncertainty)
        return output_path

    def retrieve_pairs(self, query_global: Path, query_name: str, pairs_path: Path) -> Path:
        with h5py.File(str(query_global), "r", libver="latest") as fd:
            query_desc = torch.from_numpy(
                fd[query_name]["global_descriptor"].__array__()
            ).float()[None]
        sim = torch.einsum("id,jd->ij", query_desc.to(self.device), self.db_desc)
        _, indices = torch.topk(sim, self.top_k, dim=1)
        db_names = [self.db_names[int(index)] for index in indices[0].detach().cpu().numpy()]
        pairs_path.write_text(
            "\n".join(f"{query_name} {db_name}" for db_name in db_names),
            encoding="utf-8",
        )
        return pairs_path

    @torch.no_grad()
    def match_pairs(self, pairs_path: Path, query_local: Path, matches_path: Path) -> Path:
        pairs = []
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                qname, rname = line.split()
                pairs.append((qname, rname))
        if matches_path.exists():
            matches_path.unlink()

        dataset = self.modules["FeaturePairsDataset"](
            pairs,
            query_local,
            self.ref_local_features,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            pin_memory=self.device == "cuda",
        )
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        for index, data in enumerate(loader):
            data = {
                key: value if key.startswith("image") else value.to(self.device)
                for key, value in data.items()
            }
            if self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = self.matcher(data)
            else:
                pred = self.matcher(data)
            pair = self.modules["names_to_pair"](*pairs[index])
            with h5py.File(str(matches_path), "a", libver="latest") as fd:
                if pair in fd:
                    del fd[pair]
                group = fd.create_group(pair)
                group.create_dataset(
                    "matches0",
                    data=pred["matches0"][0].cpu().short().numpy(),
                )
                if "matching_scores0" in pred:
                    group.create_dataset(
                        "matching_scores0",
                        data=pred["matching_scores0"][0].cpu().half().numpy(),
                    )
        return matches_path

    @staticmethod
    def parse_result(results_path: Path, query_name: str):
        if not results_path.exists():
            return None
        for line in results_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts and parts[0] == query_name:
                return [float(value) for value in parts[1:5]], [
                    float(value) for value in parts[5:8]
                ]
        return None

    def localize_image(self, image_path: Path, query_line: str, query_id: str) -> dict:
        image_path = image_path.resolve()
        runtime_dir = self.runtime_root / query_id
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        image_dir = runtime_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        image_name = image_path.name
        symlink_or_copy(image_path, image_dir / image_name)
        queries_path = runtime_dir / "queries_with_intrinsics.txt"
        queries_path.write_text(query_line + "\n", encoding="utf-8")

        times = {}
        total_start = time.perf_counter()
        query_global, times["netvlad_s"] = self.timed(
            lambda: self.extract_one(
                self.netvlad,
                self.retrieval_conf,
                image_dir,
                image_name,
                runtime_dir / f"{self.retrieval_conf['output']}.h5",
            )
        )
        query_local, times["superpoint_s"] = self.timed(
            lambda: self.extract_one(
                self.superpoint,
                self.local_conf,
                image_dir,
                image_name,
                runtime_dir / f"{self.local_conf['output']}.h5",
            )
        )
        pairs_path, times["retrieval_s"] = self.timed(
            lambda: self.retrieve_pairs(query_global, image_name, runtime_dir / "pairs.txt")
        )
        matches_path, times["match_s"] = self.timed(
            lambda: self.match_pairs(pairs_path, query_local, runtime_dir / "matches.h5")
        )
        results_path = runtime_dir / "localization.txt"
        _, times["localize_s"] = self.timed(
            lambda: self.modules["localize_sfm"].main(
                self.reconstruction,
                queries_path,
                pairs_path,
                query_local,
                matches_path,
                results_path,
                ransac_thresh=self.ransac_thresh,
            )
        )

        result = self.parse_result(results_path, image_name)
        num_inliers = self.read_num_inliers(runtime_dir / "localization.txt_logs.pkl", image_name)
        confidence = min(float(num_inliers) / self.confidence_scale_inliers, 1.0)
        payload = {
            "query_name": image_name,
            "query_image": str(image_path),
            "success": result is not None,
            "accepted": False,
            "status": "not_localized",
            "num_inliers": num_inliers,
            "confidence": confidence,
            "times": times,
            "runtime_dir": str(runtime_dir),
        }

        if result is not None:
            qvec, tvec = result
            center_map, quaternion_xyzw = self.calibration.apply_camera_pose(qvec, tvec)
            payload.update(
                {
                    "qvec_wxyz_raw": qvec,
                    "tvec_cam_from_world_raw": tvec,
                    "camera_center_map": center_map.tolist(),
                    "quaternion_xyzw_map": quaternion_xyzw.tolist(),
                }
            )
            if num_inliers >= self.min_inliers:
                payload["accepted"] = True
                payload["status"] = "localized"
            else:
                payload["status"] = "low_confidence"

        times["total_s"] = time.perf_counter() - total_start
        return payload

    @staticmethod
    def timed(callback):
        started = time.perf_counter()
        result = callback()
        return result, time.perf_counter() - started

    @staticmethod
    def read_num_inliers(logs_path: Path, query_name: str) -> int:
        if not logs_path.exists():
            return 0
        logs = pickle.loads(logs_path.read_bytes())
        return int(
            logs.get("loc", {})
            .get(query_name, {})
            .get("PnP_ret", {})
            .get("num_inliers", 0)
        )


class HlocLocalizationNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.pending_frame: Optional[PendingFrame] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.last_keyframe_time = rospy.Time(0)
        self.processing = False
        self.busy_drop_count = 0
        self.worker_error_logged = False

        self.image_topic = str(rospy.get_param("~image_topic", "/arducam_imx219/image_raw"))
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", ""))
        self.pose_topic = str(rospy.get_param("~pose_topic", "/planner/current_pose"))
        self.status_topic = str(rospy.get_param("~status_topic", "/hloc_localization/status"))
        self.frame_id = str(rospy.get_param("~frame_id", "map"))

        self.reference_dir = self.resolve_reference_dir(rospy.get_param("~reference_dir", ""))
        self.calibration_file = self.resolve_calibration_file(
            rospy.get_param("~calibration_file", "")
        )
        self.runtime_dir = self.resolve_runtime_dir(rospy.get_param("~runtime_dir", ""))
        self.live_frames_dir = self.runtime_dir / "live_frames"
        self.live_frames_dir.mkdir(parents=True, exist_ok=True)

        self.min_interval = float(rospy.get_param("~min_interval", 0.25))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95))
        self.blur_threshold = float(rospy.get_param("~blur_threshold", 45.0))
        self.top_k = int(rospy.get_param("~top_k", rospy.get_param("~num_matched", 10)))
        self.max_keypoints = int(rospy.get_param("~max_keypoints", 1024))
        self.resize_max = int(rospy.get_param("~resize_max", 1600))
        self.matcher = str(rospy.get_param("~matcher", "superpoint+lightglue"))
        self.ransac_thresh = int(rospy.get_param("~ransac_thresh", 12))
        self.min_inliers = int(rospy.get_param("~min_inliers", 80))
        self.device = str(rospy.get_param("~device", "auto"))
        self.fp16 = self.get_bool_param("~fp16", True)
        self.confidence_scale_inliers = float(
            rospy.get_param("~confidence_scale_inliers", 200.0)
        )
        self.result_log_enabled = self.get_bool_param("~enable_result_log", False)
        self.result_log_path = (
            self.resolve_result_log_path(rospy.get_param("~result_log_path", ""))
            if self.result_log_enabled
            else None
        )

        self.fx_param = float(rospy.get_param("~fx", 0.0))
        self.fy_param = float(rospy.get_param("~fy", 0.0))
        self.cx_param = float(rospy.get_param("~cx", 0.0))
        self.cy_param = float(rospy.get_param("~cy", 0.0))
        self.prefer_camera_info_distortion = self.get_bool_param(
            "~prefer_camera_info_distortion",
            True,
        )

        torch_home = str(rospy.get_param("~torch_home", "")).strip()
        if torch_home:
            os.environ["TORCH_HOME"] = str(Path(torch_home).expanduser())
        hloc_third_party_dir = str(rospy.get_param("~hloc_third_party_dir", "")).strip()
        if hloc_third_party_dir:
            third_party_path = Path(hloc_third_party_dir).expanduser().resolve()
            if third_party_path.exists():
                sys.path.insert(0, str(third_party_path))
            else:
                rospy.logwarn("HLoc third_party path does not exist: %s", third_party_path)

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10)

        calibration = CalibrationMetadata.load(self.calibration_file)
        self.localizer = PersistentHlocLocalizer(
            reference_dir=self.reference_dir,
            calibration=calibration,
            runtime_root=self.runtime_dir / "live_worker",
            top_k=self.top_k,
            max_keypoints=self.max_keypoints,
            resize_max=self.resize_max,
            matcher_name=self.matcher,
            ransac_thresh=self.ransac_thresh,
            min_inliers=self.min_inliers,
            confidence_scale_inliers=self.confidence_scale_inliers,
            device=self.device,
            fp16=self.fp16,
        )

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
                "No camera_info_topic configured; using fallback FULL_OPENCV intrinsics."
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

    def resolve_calibration_file(self, param_value: object) -> Path:
        if isinstance(param_value, str) and param_value.strip():
            return Path(param_value).expanduser().resolve()
        return (self.reference_dir.parent / "calibration.json").resolve()

    def resolve_runtime_dir(self, param_value: object) -> Path:
        if isinstance(param_value, str) and param_value.strip():
            runtime_dir = Path(param_value).expanduser().resolve()
        else:
            runtime_dir = self.reference_dir / "runtime_queries" / "ros_live"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir

    def resolve_result_log_path(self, param_value: object) -> Optional[Path]:
        if isinstance(param_value, str) and param_value.strip():
            path = Path(param_value).expanduser().resolve()
        else:
            path = self.runtime_dir / "live_results.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def camera_info_callback(self, msg: CameraInfo) -> None:
        with self.lock:
            self.latest_camera_info = msg

    def image_callback(self, msg: Image) -> None:
        now = rospy.Time.now()
        with self.lock:
            if self.processing:
                self.busy_drop_count += 1
                if self.busy_drop_count == 1 or self.busy_drop_count % 20 == 0:
                    self.publish_status(f"frame_dropped_busy:{self.busy_drop_count}")
                return
            if (
                self.min_interval > 0.0
                and self.last_keyframe_time != rospy.Time(0)
                and (now - self.last_keyframe_time).to_sec() < self.min_interval
            ):
                return
            self.pending_frame = PendingFrame(
                msg=msg,
                camera_info=self.latest_camera_info,
            )
            self.processing = True
            self.last_keyframe_time = now
            self.frame_event.set()

    def worker_loop(self) -> None:
        while not rospy.is_shutdown():
            if not self.frame_event.wait(0.2):
                continue

            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
                self.frame_event.clear()

            try:
                if frame is not None:
                    self.process_frame(frame)
                self.worker_error_logged = False
            except Exception as error:  # noqa: BLE001 - keep node alive after bad frames.
                self.publish_status(f"failed:{error}")
                if not self.worker_error_logged:
                    rospy.logerr("HLoc localization failed: %s", error)
                    rospy.logdebug(traceback.format_exc())
                    self.worker_error_logged = True
            finally:
                with self.lock:
                    self.processing = False

    def process_frame(self, frame: PendingFrame) -> None:
        image_bgr = self.bridge.imgmsg_to_cv2(frame.msg, desired_encoding="bgr8")
        if image_bgr is None or image_bgr.size == 0:
            raise CvBridgeError("empty image")

        blur_score = self.compute_blur_score(image_bgr)
        if self.blur_threshold > 0.0 and blur_score < self.blur_threshold:
            self.publish_status(f"frame_rejected_blur:{blur_score:.2f}")
            return

        stamp_ns = self.stamp_to_ns(frame.msg.header.stamp)
        query_name = "live_query.jpg"
        query_path = self.live_frames_dir / query_name
        if not cv2.imwrite(
            str(query_path),
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        ):
            raise RuntimeError(f"failed to write query image: {query_path}")

        query_line = self.build_query_line(query_name, image_bgr, frame.camera_info)
        result = self.localizer.localize_image(query_path, query_line, "live")
        result["live_query_line"] = query_line
        result["live_blur_score"] = blur_score
        result["image_stamp_ns"] = stamp_ns
        self.publish_localization_result(result, frame.msg.header.stamp)

    @staticmethod
    def compute_blur_score(image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def stamp_to_ns(stamp: rospy.Time) -> int:
        if stamp is None or stamp == rospy.Time(0):
            stamp = rospy.Time.now()
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)

    def build_query_line(
        self,
        image_name: str,
        image_bgr: np.ndarray,
        camera_info: Optional[CameraInfo],
    ) -> str:
        if camera_info is not None:
            query_line = self.build_query_line_from_camera_info(image_name, camera_info)
            if query_line:
                return query_line

        height, width = image_bgr.shape[:2]
        if self.fx_param > 0.0 and self.fy_param > 0.0:
            cx = self.cx_param if self.cx_param > 0.0 else width * 0.5
            cy = self.cy_param if self.cy_param > 0.0 else height * 0.5
            return colmap_camera_line(
                image_name,
                "PINHOLE",
                width,
                height,
                [self.fx_param, self.fy_param, cx, cy],
            )

        return colmap_camera_line(
            image_name,
            "FULL_OPENCV",
            1920,
            1200,
            [
                1216.44616313,
                1214.18811858,
                956.646819947,
                583.900125334,
                -0.0819699141448,
                -0.03313321259,
                -0.00244342779542,
                -0.000488657078362,
                0.0165047957227,
                0.0,
                0.0,
                0.0,
            ],
        )

    def build_query_line_from_camera_info(
        self,
        image_name: str,
        camera_info: CameraInfo,
    ) -> Optional[str]:
        width = int(camera_info.width)
        height = int(camera_info.height)
        fx = float(camera_info.K[0])
        fy = float(camera_info.K[4])
        cx = float(camera_info.K[2])
        cy = float(camera_info.K[5])
        if width <= 0 or height <= 0 or fx <= 0.0 or fy <= 0.0:
            return None

        distortion = [float(value) for value in camera_info.D]
        distortion_model = str(camera_info.distortion_model).lower()
        if self.prefer_camera_info_distortion and has_nonzero_distortion(distortion):
            if distortion_model in {"plumb_bob", "rational_polynomial"}:
                k1 = distortion[0] if len(distortion) > 0 else 0.0
                k2 = distortion[1] if len(distortion) > 1 else 0.0
                p1 = distortion[2] if len(distortion) > 2 else 0.0
                p2 = distortion[3] if len(distortion) > 3 else 0.0
                k3 = distortion[4] if len(distortion) > 4 else 0.0
                k4 = distortion[5] if len(distortion) > 5 else 0.0
                k5 = distortion[6] if len(distortion) > 6 else 0.0
                k6 = distortion[7] if len(distortion) > 7 else 0.0
                if (
                    math.isclose(fx, fy, rel_tol=0.02)
                    and abs(k2) < 1e-9
                    and abs(p1) < 1e-9
                    and abs(p2) < 1e-9
                    and abs(k3) < 1e-9
                ):
                    return colmap_camera_line(
                        image_name,
                        "SIMPLE_RADIAL",
                        width,
                        height,
                        [(fx + fy) * 0.5, cx, cy, k1],
                    )
                if has_nonzero_distortion([k3, k4, k5, k6]):
                    return colmap_camera_line(
                        image_name,
                        "FULL_OPENCV",
                        width,
                        height,
                        [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6],
                    )
                return colmap_camera_line(
                    image_name,
                    "OPENCV",
                    width,
                    height,
                    [fx, fy, cx, cy, k1, k2, p1, p2],
                )
            if distortion_model in {"equidistant", "fisheye"}:
                coeffs = [0.0, 0.0, 0.0, 0.0]
                for index, value in enumerate(distortion[:4]):
                    coeffs[index] = value
                return colmap_camera_line(
                    image_name,
                    "OPENCV_FISHEYE",
                    width,
                    height,
                    [fx, fy, cx, cy, *coeffs],
                )

        return colmap_camera_line(
            image_name,
            "PINHOLE",
            width,
            height,
            [fx, fy, cx, cy],
        )

    def publish_localization_result(self, result: dict, image_stamp: rospy.Time) -> None:
        status = str(result.get("status", "unknown"))
        num_inliers = int(result.get("num_inliers", 0))
        confidence = float(result.get("confidence", 0.0))
        total_s = float(result.get("times", {}).get("total_s", 0.0))
        self.publish_status(
            f"{status}:inliers={num_inliers}:confidence={confidence:.3f}:total_s={total_s:.3f}"
        )
        self.append_result_log(result)

        if not bool(result.get("accepted", False)):
            return

        center = result["camera_center_map"]
        quat = result["quaternion_xyzw_map"]
        pose = PoseStamped()
        pose.header.stamp = image_stamp if image_stamp and image_stamp != rospy.Time(0) else rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(center[0])
        pose.pose.position.y = float(center[1])
        pose.pose.position.z = float(center[2])
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(pose)
        rospy.loginfo_throttle(
            2.0,
            "HLoc pose published: xyz=[%.3f, %.3f, %.3f] inliers=%d confidence=%.3f",
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            num_inliers,
            confidence,
        )

    def append_result_log(self, result: dict) -> None:
        if self.result_log_path is None:
            return
        record = {
            "query_name": result.get("query_name"),
            "status": result.get("status"),
            "accepted": result.get("accepted"),
            "num_inliers": result.get("num_inliers"),
            "confidence": result.get("confidence"),
            "times": result.get("times", {}),
            "camera_center_map": result.get("camera_center_map"),
            "quaternion_xyzw_map": result.get("quaternion_xyzw_map"),
            "runtime_dir": result.get("runtime_dir"),
            "query_image": result.get("query_image"),
            "live_query_line": result.get("live_query_line"),
            "live_blur_score": result.get("live_blur_score"),
            "image_stamp_ns": result.get("image_stamp_ns"),
        }
        with self.result_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")

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
