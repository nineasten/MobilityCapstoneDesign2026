#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Path as RosPath


@dataclass(frozen=True)
class CameraModel:
    position: np.ndarray
    yaw: float
    pitch: float
    fx: float
    fy: float
    cx: float
    cy: float


def optional_param(name: str, default: object | None = None) -> object | None:
    if rospy.has_param(name):
        return rospy.get_param(name)
    return default


def has_value(value: object | None) -> bool:
    return value is not None and str(value).strip() != ""


def point_to_array(point: Point, floor_z: float) -> np.ndarray:
    return np.array([point.x, point.y, floor_z], dtype=np.float64)


def path_points(path_msg: RosPath, floor_z: float) -> list[np.ndarray]:
    return [
        point_to_array(pose.pose.position, floor_z)
        for pose in path_msg.poses
    ]


def auto_yaw(points: list[np.ndarray], lookahead_distance: float) -> float:
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


def camera_basis(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def project(point: np.ndarray, camera: CameraModel) -> tuple[int, int] | None:
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


def make_camera(points: list[np.ndarray], image_shape: tuple[int, int, int]) -> CameraModel:
    height, width = image_shape[:2]
    floor_z = float(rospy.get_param("~floor_z", 0.03))
    camera_height = float(rospy.get_param("~camera_height", 1.55))
    lookahead_distance = float(rospy.get_param("~lookahead_distance", 2.0))

    camera_x = optional_param("~camera_x")
    camera_y = optional_param("~camera_y")
    camera_z = optional_param("~camera_z")
    if camera_x is None or camera_y is None or camera_z is None:
        start = points[0] if points else np.zeros(3, dtype=np.float64)
        position = np.array(
            [start[0], start[1], floor_z + camera_height],
            dtype=np.float64,
        )
    else:
        position = np.array(
            [float(camera_x), float(camera_y), float(camera_z)],
            dtype=np.float64,
        )

    yaw_param = optional_param("~camera_yaw_deg")
    if not has_value(yaw_param):
        yaw = auto_yaw(points, lookahead_distance)
        yaw_source = "auto path heading"
    else:
        yaw = math.radians(float(yaw_param))
        yaw_source = "camera_yaw_deg"

    pitch = math.radians(float(rospy.get_param("~camera_pitch_deg", -10.0)))
    principal_x = float(rospy.get_param("~principal_x", width * 0.5))
    principal_y = float(rospy.get_param("~principal_y", height * 0.5))

    focal_param = optional_param("~focal_length_px")
    if focal_param is None:
        horizontal_fov = math.radians(
            float(rospy.get_param("~horizontal_fov_deg", 70.0))
        )
        focal = (width * 0.5) / math.tan(horizontal_fov * 0.5)
    else:
        focal = float(focal_param)

    rospy.loginfo(
        "Path overlay camera: xyz=[%.3f, %.3f, %.3f], yaw=%.1f deg (%s), pitch=%.1f deg, focal=%.1f px",
        position[0],
        position[1],
        position[2],
        math.degrees(yaw),
        yaw_source,
        math.degrees(pitch),
        focal,
    )
    return CameraModel(
        position=position,
        yaw=yaw,
        pitch=pitch,
        fx=focal,
        fy=focal,
        cx=principal_x,
        cy=principal_y,
    )


def draw_path(image: np.ndarray, points: list[np.ndarray], camera: CameraModel) -> np.ndarray:
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

    rospy.loginfo(
        "Projected %d path points with %d visible line segments.",
        len(points),
        valid_segments,
    )
    return output


def main() -> None:
    rospy.init_node("path_image_overlay")

    image_path_param = str(rospy.get_param("~image_path", "")).strip()
    if not image_path_param:
        rospy.logerr("~image_path is required.")
        sys.exit(2)

    image_path = Path(image_path_param).expanduser()
    output_path_param = str(rospy.get_param("~output_path", "")).strip()
    if output_path_param:
        output_path = Path(output_path_param).expanduser()
    else:
        output_path = image_path.with_name(f"{image_path.stem}_path_overlay.png")
    path_topic = str(rospy.get_param("~path_topic", "/path_planner/path"))
    timeout = float(rospy.get_param("~wait_timeout", 10.0))
    floor_z = float(rospy.get_param("~floor_z", 0.03))

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        rospy.logerr("Could not read image: %s", image_path)
        sys.exit(2)

    rospy.loginfo("Waiting for path on %s", path_topic)
    try:
        path_msg = rospy.wait_for_message(path_topic, RosPath, timeout=timeout)
    except rospy.ROSException as error:
        rospy.logerr("Timed out waiting for %s: %s", path_topic, error)
        sys.exit(1)

    points = path_points(path_msg, floor_z)
    if len(points) < 2:
        rospy.logerr("Path needs at least 2 points; got %d.", len(points))
        sys.exit(1)

    camera = make_camera(points, image.shape)
    output = draw_path(image, points, camera)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        rospy.logerr("Failed to write overlay image: %s", output_path)
        sys.exit(1)
    rospy.loginfo("Wrote path overlay image: %s", output_path)


if __name__ == "__main__":
    main()
