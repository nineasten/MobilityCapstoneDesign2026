#!/usr/bin/env python3
"""Publish a simple RViz marker for the latest HLOC PoseStamped."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

import rospy
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def color(r: float, g: float, b: float, a: float) -> ColorRGBA:
    msg = ColorRGBA()
    msg.r = r
    msg.g = g
    msg.b = b
    msg.a = a
    return msg


def rotate_vector_by_quaternion(q, vector: tuple[float, float, float]) -> Point:
    vx, vy, vz = vector
    qx = q.x
    qy = q.y
    qz = q.z
    qw = q.w

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return Point(
        x=vx + qw * tx + (qy * tz - qz * ty),
        y=vy + qw * ty + (qz * tx - qx * tz),
        z=vz + qw * tz + (qx * ty - qy * tx),
    )


class HlocPoseMarkerNode:
    def __init__(self) -> None:
        self.pose_topic = str(rospy.get_param("~pose_topic", "/planner/current_pose"))
        self.marker_topic = str(
            rospy.get_param("~marker_topic", "/hloc_localization/pose_marker")
        )
        self.frame_id = str(rospy.get_param("~frame_id", "map"))
        self.project_to_ground = bool(rospy.get_param("~project_to_ground", True))
        self.marker_z = float(rospy.get_param("~marker_z", 0.12))
        self.arrow_length = float(rospy.get_param("~arrow_length", 0.85))
        self.arrow_shaft = float(rospy.get_param("~arrow_shaft", 0.08))
        self.arrow_head = float(rospy.get_param("~arrow_head", 0.20))
        self.sphere_diameter = float(rospy.get_param("~sphere_diameter", 0.34))
        self.trail_width = float(rospy.get_param("~trail_width", 0.08))
        self.trail_size = int(rospy.get_param("~trail_size", 120))
        self.heading_axis = str(rospy.get_param("~heading_axis", "camera_z"))

        self.trail = deque(maxlen=max(1, self.trail_size))  # type: Deque[Point]
        self.publisher = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1, latch=True
        )
        self.subscriber = rospy.Subscriber(
            self.pose_topic, PoseStamped, self.pose_callback, queue_size=1
        )

    def pose_callback(self, msg: PoseStamped) -> None:
        frame_id = msg.header.frame_id or self.frame_id
        z = self.marker_z if self.project_to_ground else msg.pose.position.z
        x = msg.pose.position.x
        y = msg.pose.position.y
        heading = self.heading_vector(msg.pose.orientation)

        current = Point(x=x, y=y, z=z)
        self.trail.append(current)

        stamp = rospy.Time.now()
        marker_array = MarkerArray()
        marker_array.markers.append(self.make_delete_all(frame_id, stamp))
        marker_array.markers.append(self.make_position_marker(frame_id, stamp, current))
        marker_array.markers.append(
            self.make_heading_marker(frame_id, stamp, current, heading)
        )
        marker_array.markers.append(self.make_trail_marker(frame_id, stamp))
        self.publisher.publish(marker_array)

    def heading_vector(self, orientation) -> Point:
        axis = {
            "camera_x": (1.0, 0.0, 0.0),
            "camera_y": (0.0, 1.0, 0.0),
            "camera_z": (0.0, 0.0, 1.0),
            "camera_minus_x": (-1.0, 0.0, 0.0),
            "camera_minus_y": (0.0, -1.0, 0.0),
            "camera_minus_z": (0.0, 0.0, -1.0),
        }.get(self.heading_axis, (0.0, 0.0, 1.0))

        heading = rotate_vector_by_quaternion(orientation, axis)
        norm_xy = math.hypot(heading.x, heading.y)
        if norm_xy <= 1e-9:
            return Point(x=1.0, y=0.0, z=0.0)
        heading.x /= norm_xy
        heading.y /= norm_xy
        heading.z = 0.0
        return heading

    def base_marker(self, frame_id: str, stamp: rospy.Time, marker_id: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "hloc_pose"
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def make_delete_all(self, frame_id: str, stamp: rospy.Time) -> Marker:
        marker = self.base_marker(frame_id, stamp, 0)
        marker.action = Marker.DELETEALL
        return marker

    def make_position_marker(
        self, frame_id: str, stamp: rospy.Time, point: Point
    ) -> Marker:
        marker = self.base_marker(frame_id, stamp, 1)
        marker.type = Marker.SPHERE
        marker.pose.position = point
        marker.scale.x = self.sphere_diameter
        marker.scale.y = self.sphere_diameter
        marker.scale.z = self.sphere_diameter
        marker.color = color(0.05, 0.95, 0.35, 0.95)
        return marker

    def make_heading_marker(
        self, frame_id: str, stamp: rospy.Time, point: Point, heading: Point
    ) -> Marker:
        marker = self.base_marker(frame_id, stamp, 2)
        marker.type = Marker.ARROW
        marker.points.append(point)
        marker.points.append(
            Point(
                x=point.x + self.arrow_length * heading.x,
                y=point.y + self.arrow_length * heading.y,
                z=point.z,
            )
        )
        marker.scale.x = self.arrow_shaft
        marker.scale.y = self.arrow_head
        marker.scale.z = self.arrow_head
        marker.color = color(0.05, 0.95, 0.35, 0.95)
        return marker

    def make_trail_marker(self, frame_id: str, stamp: rospy.Time) -> Marker:
        marker = self.base_marker(frame_id, stamp, 3)
        marker.type = Marker.LINE_STRIP
        marker.points = list(self.trail)
        marker.scale.x = self.trail_width
        marker.color = color(0.15, 0.55, 1.0, 0.80)
        return marker


def main() -> None:
    rospy.init_node("hloc_pose_marker")
    HlocPoseMarkerNode()
    rospy.loginfo("HLOC pose marker ready")
    rospy.spin()


if __name__ == "__main__":
    main()
