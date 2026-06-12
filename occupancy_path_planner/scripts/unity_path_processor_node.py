#!/usr/bin/env python3

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header


@dataclass(frozen=True)
class PathPoint:
    x: float
    y: float
    z: float


class UnityPathProcessorNode:
    """Turns the planner's grid path into a smoother Unity display path."""

    def __init__(self) -> None:
        self.input_path_topic = rospy.get_param(
            "~input_path_topic", "/path_planner/path"
        )
        self.output_path_topic = rospy.get_param(
            "~output_path_topic", "/path_planner/unity_path"
        )

        self.min_input_spacing = float(rospy.get_param("~min_input_spacing", 0.05))
        self.resample_spacing = float(rospy.get_param("~resample_spacing", 0.30))
        self.smoothing_iterations = int(
            rospy.get_param("~smoothing_iterations", 0)
        )
        self.corner_cut = float(rospy.get_param("~corner_cut", 0.12))
        self.max_points = int(rospy.get_param("~max_points", 260))
        self.simplify_tolerance = float(rospy.get_param("~simplify_tolerance", 0.25))
        self.use_fixed_height = self.get_bool_param("~use_fixed_height", True)
        self.hover_height = float(rospy.get_param("~hover_height", 0.0))
        self.relative_height = self.get_bool_param("~relative_height", False)

        self.publish_lock = threading.Lock()
        self.path_pub = rospy.Publisher(
            self.output_path_topic, Path, queue_size=1, latch=True
        )
        self.path_sub = rospy.Subscriber(
            self.input_path_topic,
            Path,
            self.path_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Unity path processor listening to %s and publishing %s",
            self.input_path_topic,
            self.output_path_topic,
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

    def path_callback(self, msg: Path) -> None:
        points = self.extract_points(msg)
        if len(points) < 2:
            self.publish_path([], msg)
            return

        processed = self.process_path(points)
        self.publish_path(processed, msg)
        rospy.loginfo_throttle(
            5.0,
            "Published Unity path with %d points from raw %d points",
            len(processed),
            len(points),
        )

    def extract_points(self, msg: Path) -> list[PathPoint]:
        points: list[PathPoint] = []
        for pose_stamped in msg.poses:
            position = pose_stamped.pose.position
            point = PathPoint(
                float(position.x),
                float(position.y),
                float(position.z),
            )
            if not points or self.xy_distance(points[-1], point) >= self.min_input_spacing:
                points.append(point)
        return points

    def process_path(self, points: list[PathPoint]) -> list[PathPoint]:
        simplified = self.douglas_peucker(points, max(0.0, self.simplify_tolerance))

        smoothed = simplified[:]
        iterations = max(0, self.smoothing_iterations)
        corner_cut = min(max(self.corner_cut, 0.0), 0.5)
        for _ in range(iterations):
            smoothed = self.chaikin_smooth(smoothed, corner_cut)

        spacing = max(self.resample_spacing, 0.01)
        total_length = self.path_length(smoothed)
        if self.max_points > 1 and total_length / spacing > self.max_points - 1:
            spacing = total_length / float(self.max_points - 1)

        resampled = self.resample(smoothed, spacing)
        return [self.with_display_height(point) for point in resampled]

    @staticmethod
    def douglas_peucker(points: list[PathPoint], tolerance: float) -> list[PathPoint]:
        """Remove points within tolerance distance of the straight line, keeping
        only waypoints where the path actually changes direction."""
        if tolerance <= 0.0 or len(points) < 3:
            return points[:]

        def point_to_segment_dist(p: PathPoint, a: PathPoint, b: PathPoint) -> float:
            dx = b.x - a.x
            dy = b.y - a.y
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-12:
                return math.hypot(p.x - a.x, p.y - a.y)
            t = max(0.0, min(1.0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length_sq))
            return math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy))

        def simplify(pts: list[PathPoint]) -> list[PathPoint]:
            if len(pts) < 3:
                return pts[:]
            max_dist = 0.0
            max_idx = 1
            for i in range(1, len(pts) - 1):
                d = point_to_segment_dist(pts[i], pts[0], pts[-1])
                if d > max_dist:
                    max_dist = d
                    max_idx = i
            if max_dist > tolerance:
                left = simplify(pts[: max_idx + 1])
                right = simplify(pts[max_idx:])
                return left[:-1] + right
            return [pts[0], pts[-1]]

        return simplify(points)

    @staticmethod
    def chaikin_smooth(
        points: list[PathPoint],
        corner_cut: float,
    ) -> list[PathPoint]:
        if len(points) < 3 or corner_cut <= 0.0:
            return points[:]

        smoothed = [points[0]]
        for index in range(len(points) - 1):
            start = points[index]
            end = points[index + 1]
            q = UnityPathProcessorNode.interpolate(start, end, corner_cut)
            r = UnityPathProcessorNode.interpolate(start, end, 1.0 - corner_cut)
            smoothed.extend([q, r])
        smoothed.append(points[-1])
        return smoothed

    def resample(self, points: list[PathPoint], spacing: float) -> list[PathPoint]:
        if len(points) < 2:
            return points[:]

        total_length = self.path_length(points)
        if total_length <= 0.001:
            return [points[0], points[-1]]

        samples = [points[0]]
        target_distance = spacing
        segment_start_index = 0
        distance_before_segment = 0.0

        while target_distance < total_length:
            while segment_start_index < len(points) - 1:
                start = points[segment_start_index]
                end = points[segment_start_index + 1]
                segment_length = self.xy_distance(start, end)
                if distance_before_segment + segment_length >= target_distance:
                    ratio = (
                        (target_distance - distance_before_segment) / segment_length
                        if segment_length > 0.0
                        else 0.0
                    )
                    samples.append(self.interpolate(start, end, ratio))
                    break

                distance_before_segment += segment_length
                segment_start_index += 1

            target_distance += spacing

        if self.xy_distance(samples[-1], points[-1]) > 0.001:
            samples.append(points[-1])
        return samples

    def with_display_height(self, point: PathPoint) -> PathPoint:
        if self.use_fixed_height:
            z = self.hover_height
        elif self.relative_height:
            z = point.z + self.hover_height
        else:
            z = point.z
        return PathPoint(point.x, point.y, z)

    def publish_path(self, points: list[PathPoint], source_msg: Path) -> None:
        path = Path()
        path.header = Header(
            stamp=rospy.Time.now(),
            frame_id=source_msg.header.frame_id,
        )

        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = Point(point.x, point.y, point.z)
            pose.pose.orientation = self.orientation_for_point(points, index)
            path.poses.append(pose)

        with self.publish_lock:
            self.path_pub.publish(path)

    def orientation_for_point(
        self,
        points: list[PathPoint],
        index: int,
    ) -> Quaternion:
        if len(points) < 2:
            return Quaternion(w=1.0)

        if index == 0:
            start = points[0]
            end = points[1]
        elif index == len(points) - 1:
            start = points[index - 1]
            end = points[index]
        else:
            start = points[index - 1]
            end = points[index + 1]

        yaw = math.atan2(end.y - start.y, end.x - start.x)
        return self.quaternion_from_yaw(yaw)

    @staticmethod
    def quaternion_from_yaw(yaw: float) -> Quaternion:
        half_yaw = yaw * 0.5
        return Quaternion(z=math.sin(half_yaw), w=math.cos(half_yaw))

    @staticmethod
    def interpolate(start: PathPoint, end: PathPoint, ratio: float) -> PathPoint:
        clamped = min(max(ratio, 0.0), 1.0)
        return PathPoint(
            start.x + (end.x - start.x) * clamped,
            start.y + (end.y - start.y) * clamped,
            start.z + (end.z - start.z) * clamped,
        )

    @staticmethod
    def xy_distance(start: PathPoint, end: PathPoint) -> float:
        return math.hypot(end.x - start.x, end.y - start.y)

    def path_length(self, points: list[PathPoint]) -> float:
        return sum(
            self.xy_distance(points[index - 1], points[index])
            for index in range(1, len(points))
        )

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("unity_path_processor")
    node = UnityPathProcessorNode()
    node.spin()


if __name__ == "__main__":
    main()
