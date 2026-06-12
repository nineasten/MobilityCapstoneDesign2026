#!/usr/bin/env python3

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import CameraInfo, Image


@dataclass(frozen=True)
class CameraModel:
    position: np.ndarray
    yaw: float
    pitch: float
    fx: float
    fy: float
    cx: float
    cy: float


def get_bool_param(name: str, default: bool) -> bool:
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def yaw_from_quaternion(quaternion: Quaternion) -> float:
    sin_yaw = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def point_to_array(point: Point, fallback_z: float, use_message_z: bool) -> np.ndarray:
    z = point.z if use_message_z else fallback_z
    return np.array([point.x, point.y, z], dtype=np.float64)


def path_points(
    path_msg: RosPath,
    fallback_z: float,
    use_message_z: bool,
) -> List[np.ndarray]:
    return [
        point_to_array(pose.pose.position, fallback_z, use_message_z)
        for pose in path_msg.poses
    ]


def auto_yaw(points: List[np.ndarray], lookahead_distance: float) -> float:
    if len(points) < 2:
        return 0.0

    origin = points[0]
    previous = origin
    distance = 0.0
    target = points[-1]

    for point in points[1:]:
        distance += float(np.linalg.norm(point[:2] - previous[:2]))
        target = point
        if distance >= lookahead_distance:
            break
        previous = point

    direction = target[:2] - origin[:2]
    if float(np.linalg.norm(direction)) < 1e-6:
        return 0.0
    return math.atan2(float(direction[1]), float(direction[0]))


def camera_basis(yaw: float, pitch: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.array(
        [
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)

    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)

    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return right, down, forward


def apply_camera_offset(position: np.ndarray, yaw: float) -> np.ndarray:
    forward_offset = float(rospy.get_param("~camera_forward_offset", 0.0))
    right_offset = float(rospy.get_param("~camera_right_offset", 0.0))
    up_offset = float(rospy.get_param("~camera_up_offset", 0.0))

    forward_xy = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    right_xy = np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
    return (
        position
        + forward_offset * forward_xy
        + right_offset * right_xy
        + np.array([0.0, 0.0, up_offset], dtype=np.float64)
    )


def project(point: np.ndarray, camera: CameraModel) -> Optional[Tuple[int, int]]:
    right, down, forward = camera_basis(camera.yaw, camera.pitch)
    relative = point - camera.position
    x_camera = float(np.dot(relative, right))
    y_camera = float(np.dot(relative, down))
    z_camera = float(np.dot(relative, forward))

    if z_camera <= 0.05:
        return None

    u = camera.fx * x_camera / z_camera + camera.cx
    v = camera.fy * y_camera / z_camera + camera.cy
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return int(round(u)), int(round(v))


def intrinsics_from_camera_info(
    camera_info: CameraInfo,
) -> Optional[Tuple[float, float, float, float]]:
    if len(camera_info.K) < 6:
        return None

    fx = float(camera_info.K[0])
    fy = float(camera_info.K[4])
    cx = float(camera_info.K[2])
    cy = float(camera_info.K[5])
    if fx <= 0.0 or fy <= 0.0:
        return None
    return fx, fy, cx, cy


def optional_param(name: str):
    if rospy.has_param(name):
        return rospy.get_param(name)
    return None


def intrinsics_from_params(
    image_shape: Tuple[int, int, int],
    camera_info: Optional[CameraInfo],
) -> Tuple[float, float, float, float, str]:
    if camera_info is not None:
        intrinsics = intrinsics_from_camera_info(camera_info)
        if intrinsics is not None:
            return intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], "camera_info"

    height, width = image_shape[:2]
    principal_x = float(rospy.get_param("~principal_x", width * 0.5))
    principal_y = float(rospy.get_param("~principal_y", height * 0.5))

    focal_param = optional_param("~focal_length_px")
    if focal_param is None:
        horizontal_fov = math.radians(float(rospy.get_param("~horizontal_fov_deg", 70.0)))
        focal = (width * 0.5) / math.tan(horizontal_fov * 0.5)
        return focal, focal, principal_x, principal_y, "horizontal_fov_deg"

    focal = float(focal_param)
    return focal, focal, principal_x, principal_y, "focal_length_px"


def make_camera(
    points: List[np.ndarray],
    image_shape: Tuple[int, int, int],
    pose_msg: Optional[PoseStamped],
    camera_info: Optional[CameraInfo],
) -> CameraModel:
    floor_z = float(rospy.get_param("~floor_z", 0.03))
    camera_height = float(rospy.get_param("~camera_height", 1.55))
    lookahead_distance = float(rospy.get_param("~lookahead_distance", 2.0))
    pitch = math.radians(float(rospy.get_param("~camera_pitch_deg", -12.0)))
    yaw_offset = math.radians(float(rospy.get_param("~camera_yaw_offset_deg", 0.0)))
    force_camera_height = get_bool_param("~force_camera_height", True)

    if pose_msg is not None:
        pose = pose_msg.pose
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )
        if force_camera_height:
            position[2] = floor_z + camera_height
        yaw = yaw_from_quaternion(pose.orientation) + yaw_offset
        source = "pose"
    else:
        start = points[0] if points else np.zeros(3, dtype=np.float64)
        position = np.array(
            [start[0], start[1], floor_z + camera_height],
            dtype=np.float64,
        )
        yaw = auto_yaw(points, lookahead_distance) + yaw_offset
        source = "path_start_fallback"

    position = apply_camera_offset(position, yaw)
    fx, fy, cx, cy, intrinsics_source = intrinsics_from_params(image_shape, camera_info)
    rospy.loginfo_throttle(
        5.0,
        (
            "Path overlay camera source=%s xyz=[%.3f, %.3f, %.3f], "
            "yaw=%.1f deg, pitch=%.1f deg, intrinsics=%s fx=%.1f fy=%.1f"
        ),
        source,
        position[0],
        position[1],
        position[2],
        math.degrees(yaw),
        math.degrees(pitch),
        intrinsics_source,
        fx,
        fy,
    )
    return CameraModel(
        position=position,
        yaw=yaw,
        pitch=pitch,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def draw_path(image: np.ndarray, points: List[np.ndarray], camera: CameraModel) -> np.ndarray:
    projected = [project(point, camera) for point in points]
    output = image.copy()

    shadow_width = int(rospy.get_param("~shadow_width_px", 18))
    line_width = int(rospy.get_param("~line_width_px", 8))
    arrow_stride = max(1, int(rospy.get_param("~arrow_stride_points", 8)))
    path_color = tuple(int(v) for v in rospy.get_param("~path_color_bgr", [60, 255, 80]))
    shadow_color = tuple(int(v) for v in rospy.get_param("~shadow_color_bgr", [0, 0, 0]))

    valid_segments = 0
    for start, end in zip(projected[:-1], projected[1:]):
        if start is None or end is None:
            continue
        valid_segments += 1
        cv2.line(output, start, end, shadow_color, shadow_width, cv2.LINE_AA)

    for start, end in zip(projected[:-1], projected[1:]):
        if start is None or end is None:
            continue
        cv2.line(output, start, end, path_color, line_width, cv2.LINE_AA)

    for index in range(0, len(projected) - 1, arrow_stride):
        start = projected[index]
        end = projected[index + 1]
        if start is None or end is None:
            continue
        cv2.arrowedLine(
            output,
            start,
            end,
            path_color,
            max(2, line_width - 2),
            cv2.LINE_AA,
            tipLength=0.35,
        )

    visible_points = [point for point in projected if point is not None]
    if visible_points:
        cv2.circle(output, visible_points[0], 12, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, visible_points[0], 7, (80, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(output, visible_points[-1], 14, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, visible_points[-1], 9, (0, 80, 255), -1, cv2.LINE_AA)

    rospy.loginfo_throttle(
        2.0,
        "Projected %d path points with %d visible line segments.",
        len(points),
        valid_segments,
    )
    return output


class LivePathImageOverlay:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.state_lock = threading.Lock()
        self.latest_points = []  # type: List[np.ndarray]
        self.latest_pose = None  # type: Optional[PoseStamped]
        self.latest_camera_info = None  # type: Optional[CameraInfo]

        self.image_topic = str(rospy.get_param("~image_topic", "/arducam_imx219/image_raw"))
        self.path_topic = str(rospy.get_param("~path_topic", "/path_planner/path"))
        self.pose_topic = str(rospy.get_param("~pose_topic", "/planner/current_pose"))
        self.camera_info_topic = str(
            rospy.get_param("~camera_info_topic", "/arducam_imx219/camera_info")
        ).strip()
        self.overlay_topic = str(
            rospy.get_param("~overlay_topic", "/path_planner/overlay_image")
        )

        self.queue_size = int(rospy.get_param("~queue_size", 1))
        self.use_path_z = get_bool_param("~use_path_z", True)
        self.use_path_start_fallback = get_bool_param("~use_path_start_fallback", True)
        self.publish_passthrough = get_bool_param("~publish_passthrough", True)

        self.overlay_pub = rospy.Publisher(
            self.overlay_topic,
            Image,
            queue_size=self.queue_size,
        )
        self.path_sub = rospy.Subscriber(
            self.path_topic,
            RosPath,
            self.path_callback,
            queue_size=1,
        )
        self.pose_sub = rospy.Subscriber(
            self.pose_topic,
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=self.queue_size,
            buff_size=2**24,
        )
        self.camera_info_sub = None
        if self.camera_info_topic:
            self.camera_info_sub = rospy.Subscriber(
                self.camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
                queue_size=1,
            )

        rospy.loginfo(
            "Path image overlay live: image=%s path=%s pose=%s camera_info=%s output=%s",
            self.image_topic,
            self.path_topic,
            self.pose_topic,
            self.camera_info_topic or "(disabled)",
            self.overlay_topic,
        )

    def path_callback(self, msg: RosPath) -> None:
        floor_z = float(rospy.get_param("~floor_z", 0.03))
        points = path_points(msg, floor_z, self.use_path_z)
        with self.state_lock:
            self.latest_points = points
        rospy.loginfo_throttle(2.0, "Overlay received path with %d points.", len(points))

    def pose_callback(self, msg: PoseStamped) -> None:
        with self.state_lock:
            self.latest_pose = msg

    def camera_info_callback(self, msg: CameraInfo) -> None:
        with self.state_lock:
            self.latest_camera_info = msg

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as error:
            rospy.logerr_throttle(5.0, "Failed to convert input image: %s", error)
            return

        with self.state_lock:
            points = list(self.latest_points)
            pose_msg = self.latest_pose
            camera_info = self.latest_camera_info

        output = self.render_frame(frame, points, pose_msg, camera_info)
        if output is None:
            return

        try:
            output_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
        except CvBridgeError as error:
            rospy.logerr_throttle(5.0, "Failed to convert overlay image: %s", error)
            return

        output_msg.header = msg.header
        self.overlay_pub.publish(output_msg)

    def render_frame(
        self,
        frame: np.ndarray,
        points: List[np.ndarray],
        pose_msg: Optional[PoseStamped],
        camera_info: Optional[CameraInfo],
    ) -> Optional[np.ndarray]:
        if len(points) < 2:
            rospy.logwarn_throttle(
                5.0,
                "No usable path yet on %s; publishing camera image only.",
                self.path_topic,
            )
            return frame if self.publish_passthrough else None

        if pose_msg is None and not self.use_path_start_fallback:
            rospy.logwarn_throttle(
                5.0,
                "No pose yet on %s; publishing camera image only.",
                self.pose_topic,
            )
            return frame if self.publish_passthrough else None

        camera = make_camera(
            points,
            frame.shape,
            pose_msg if pose_msg is not None else None,
            camera_info,
        )
        return draw_path(frame, points, camera)


def main() -> None:
    rospy.init_node("path_live_image_overlay")
    LivePathImageOverlay()
    rospy.spin()


if __name__ == "__main__":
    main()
