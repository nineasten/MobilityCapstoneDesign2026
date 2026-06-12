#!/usr/bin/env python3

from __future__ import annotations

import os
import math
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import TclError, ttk
except ImportError:
    tk = None
    ttk = None
    TclError = Exception

import rospkg
import rospy
import yaml
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Vector3
from nav_msgs.msg import OccupancyGrid, Path as RosPath
from std_msgs.msg import ColorRGBA, Header, String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class Destination:
    key: str
    label: str
    point: Point


class DestinationSelectorNode:
    def __init__(self) -> None:
        default_config_path = self.resolve_default_config_path()
        self.config_path = Path(
            rospy.get_param("~config_path", str(default_config_path))
        )
        self.frame_id = rospy.get_param("~frame_id", "")
        self.occupancy_topic = rospy.get_param(
            "~occupancy_topic", "/voxel_map/xy_occupancy"
        )
        self.current_pose_topic = rospy.get_param(
            "~current_pose_topic", "/planner/current_pose"
        )
        self.current_pose_project_to_ground = self.get_bool_param(
            "~current_pose_project_to_ground", True
        )
        self.current_pose_ground_z = float(
            rospy.get_param("~current_pose_ground_z", 0.0)
        )
        self.current_pose_overrides_auto_advance = self.get_bool_param(
            "~current_pose_overrides_auto_advance", True
        )
        self.clicked_point_topic = rospy.get_param(
            "~clicked_point_topic", "/clicked_point"
        )
        self.current_point_topic = rospy.get_param(
            "~current_point_topic", "/current_position_point"
        )
        self.clicked_point_mode = str(
            rospy.get_param("~clicked_point_mode", "current")
        ).strip().lower()
        self.clicked_destination_prefix = str(
            rospy.get_param("~clicked_destination_prefix", "clicked")
        ).strip()
        self.clicked_destination_separator = str(
            rospy.get_param("~clicked_destination_separator", "_")
        )
        self.clicked_destination_index_digits = max(
            0, int(rospy.get_param("~clicked_destination_index_digits", 3))
        )
        self.clicked_destination_label_prefix = str(
            rospy.get_param("~clicked_destination_label_prefix", "Clicked")
        ).strip()
        self.clicked_destination_z = float(
            rospy.get_param("~clicked_destination_z", 0.0)
        )
        self.autosave_destinations = self.get_bool_param(
            "~autosave_destinations", True
        )
        self.show_candidate_markers = self.get_bool_param(
            "~show_candidate_markers", False
        )
        self.candidate_marker_delete_count = max(
            0, int(rospy.get_param("~candidate_marker_delete_count", 200))
        )
        self.selection_topic = rospy.get_param(
            "~selection_topic", "/destination_selector/select_goal"
        )
        self.markers_topic = rospy.get_param(
            "~markers_topic", "/destination_selector/markers"
        )
        self.selected_goal_topic = rospy.get_param(
            "~selected_goal_topic", "/destination_selector/selected_goal"
        )
        self.selected_goal_name_topic = rospy.get_param(
            "~selected_goal_name_topic",
            "/destination_selector/selected_goal_name",
        )
        self.current_position_topic = rospy.get_param(
            "~current_position_topic", "/destination_selector/current_position"
        )
        self.path_topic = rospy.get_param("~path_topic", "/path_planner/path")
        self.auto_advance_current = self.get_bool_param(
            "~auto_advance_current", False
        )
        self.auto_advance_distance = float(
            rospy.get_param("~auto_advance_distance", 5.0)
        )
        self.auto_advance_period = float(
            rospy.get_param("~auto_advance_period", 2.0)
        )
        self.auto_advance_min_move = float(
            rospy.get_param("~auto_advance_min_move", 0.1)
        )
        self.publish_rate = float(rospy.get_param("~publish_rate", 2.0))
        self.prompt_on_start = self.get_bool_param("~prompt_on_start", True)
        self.gui_on_start = self.get_bool_param("~gui_on_start", False)
        self.gui_title = str(
            rospy.get_param("~gui_title", "Destination Selector")
        )
        self.gui_refresh_ms = int(rospy.get_param("~gui_refresh_ms", 500))
        self.shutdown_on_gui_close = self.get_bool_param(
            "~shutdown_on_gui_close", False
        )
        self.rviz_on_select = self.get_bool_param("~rviz_on_select", True)
        self.rviz_config_path = Path(
            rospy.get_param(
                "~rviz_config_path", str(self.resolve_default_rviz_config_path())
            )
        )

        self.marker_z_offset = float(rospy.get_param("~marker_z_offset", 0.12))
        self.text_z_offset = float(rospy.get_param("~text_z_offset", 0.35))
        self.current_marker_scale = float(
            rospy.get_param("~current_marker_scale", 0.40)
        )
        self.goal_marker_scale = float(
            rospy.get_param("~goal_marker_scale", 0.42)
        )
        self.candidate_marker_scale = float(
            rospy.get_param("~candidate_marker_scale", 0.16)
        )

        self.state_lock = threading.Lock()
        self.current_position: Point | None = None
        self.current_position_source = "unknown"
        self.received_current_pose = False
        self.selected_destination: Destination | None = None
        self.latest_path_points: list[Point] = []
        self.latest_path_version = 0
        self.last_auto_advance_path_version = -1
        self.sequence_current_position_set = False
        self.rviz_process: subprocess.Popen | None = None

        self.destinations = self.load_destinations(self.config_path)
        if not self.frame_id:
            self.frame_id = "map"
        if self.clicked_point_mode not in {
            "current",
            "destination",
            "candidate",
            "sequence",
        }:
            raise ValueError(
                "clicked_point_mode must be 'current', 'destination', 'candidate', or 'sequence'"
            )

        self.marker_pub = rospy.Publisher(
            self.markers_topic, MarkerArray, queue_size=1, latch=True
        )
        self.selected_goal_pub = rospy.Publisher(
            self.selected_goal_topic, PointStamped, queue_size=1, latch=True
        )
        self.selected_goal_name_pub = rospy.Publisher(
            self.selected_goal_name_topic, String, queue_size=1, latch=True
        )
        self.current_position_pub = rospy.Publisher(
            self.current_position_topic, PointStamped, queue_size=1, latch=True
        )

        self.occupancy_sub = rospy.Subscriber(
            self.occupancy_topic,
            OccupancyGrid,
            self.occupancy_callback,
            queue_size=1,
        )
        self.current_pose_sub = rospy.Subscriber(
            self.current_pose_topic,
            PoseStamped,
            self.current_pose_callback,
            queue_size=1,
        )
        self.clicked_point_sub = rospy.Subscriber(
            self.clicked_point_topic,
            PointStamped,
            self.clicked_point_callback,
            queue_size=1,
        )
        self.current_point_sub = None
        if self.current_point_topic and self.current_point_topic != self.clicked_point_topic:
            self.current_point_sub = rospy.Subscriber(
                self.current_point_topic,
                PointStamped,
                self.current_point_callback,
                queue_size=1,
            )
        self.selection_sub = rospy.Subscriber(
            self.selection_topic,
            String,
            self.selection_callback,
            queue_size=1,
        )
        self.path_sub = rospy.Subscriber(
            self.path_topic,
            RosPath,
            self.path_callback,
            queue_size=1,
        )
        self.auto_advance_timer = None
        if self.auto_advance_current:
            self.auto_advance_timer = rospy.Timer(
                rospy.Duration(max(self.auto_advance_period, 0.1)),
                self.auto_advance_current_callback,
            )

        rospy.loginfo(
            "Loaded %d destinations from %s",
            len(self.destinations),
            self.config_path,
        )
        for index, destination in enumerate(self.destinations, start=1):
            rospy.loginfo(
                "  [%d] %s (%s): [%.3f, %.3f, %.3f]",
                index,
                destination.label,
                destination.key,
                destination.point.x,
                destination.point.y,
                destination.point.z,
            )

        self.publish_markers()
        self.publish_current_position()

        if self.prompt_on_start:
            prompt_thread = threading.Thread(
                target=self.prompt_loop,
                name="destination_selector_prompt",
                daemon=True,
            )
            prompt_thread.start()

    @staticmethod
    def resolve_default_config_path() -> Path:
        package_path = Path(rospkg.RosPack().get_path("destination_selector"))
        return package_path / "config" / "destinations.yaml"

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

    def load_destinations(self, path: Path) -> list[Destination]:
        if not path.exists():
            raise FileNotFoundError(f"Destination config not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        if config.get("frame_id") and not self.frame_id:
            self.frame_id = str(config["frame_id"])
        self.marker_z_offset = float(
            rospy.get_param(
                "~marker_z_offset",
                config.get("marker_z_offset", self.marker_z_offset),
            )
        )
        self.text_z_offset = float(
            rospy.get_param(
                "~text_z_offset",
                config.get("text_z_offset", self.text_z_offset),
            )
        )

        raw_current = config.get("current_position") or config.get("current")
        if isinstance(raw_current, dict):
            xyz = raw_current.get("xyz") or raw_current.get("position")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise ValueError("current_position must define xyz: [x, y, z]")
            self.current_position = Point(
                float(xyz[0]), float(xyz[1]), float(xyz[2])
            )
            self.current_position_source = "config"

        raw_destinations = config.get("destinations", {})
        destinations: list[Destination] = []

        if isinstance(raw_destinations, dict):
            items = raw_destinations.items()
        elif isinstance(raw_destinations, list):
            items = []
            for item in raw_destinations:
                key = item.get("key") or item.get("id")
                items.append((key, item))
        else:
            raise ValueError("destinations must be a mapping or a list")

        for raw_key, value in items:
            if raw_key is None:
                raise ValueError("Each destination requires a key or id")
            if not isinstance(value, dict):
                raise ValueError(f"Invalid destination entry for {raw_key}")

            label = str(value.get("label", raw_key))
            xyz = value.get("xyz") or value.get("position")
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                raise ValueError(
                    f"Destination {raw_key} must define xyz: [x, y, z]"
                )

            point = Point(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            destinations.append(
                Destination(key=str(raw_key), label=label, point=point)
            )

        if not destinations:
            raise ValueError(f"No destinations defined in {path}")

        return destinations

    def save_destinations(self) -> None:
        with self.state_lock:
            current = self.current_position
            destinations = list(self.destinations)

        payload: dict[str, object] = {
            "frame_id": self.frame_id,
            "marker_z_offset": float(self.marker_z_offset),
            "text_z_offset": float(self.text_z_offset),
            "destinations": {
                destination.key: {
                    "label": destination.label,
                    "xyz": [
                        float(destination.point.x),
                        float(destination.point.y),
                        float(destination.point.z),
                    ],
                }
                for destination in destinations
            },
        }
        if current is not None:
            payload["current_position"] = {
                "label": "Current Position",
                "xyz": [float(current.x), float(current.y), float(current.z)],
            }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                payload,
                file,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    def next_clicked_destination_key(self) -> tuple[str, int]:
        prefix = self.clicked_destination_prefix or "clicked"
        separator = self.clicked_destination_separator
        digits = self.clicked_destination_index_digits
        existing_keys = {destination.key for destination in self.destinations}
        index = 1
        while True:
            index_text = f"{index:0{digits}d}" if digits > 0 else str(index)
            key = f"{prefix}{separator}{index_text}"
            if key not in existing_keys:
                return key, index
            index += 1

    def add_destination_from_clicked_point(self, msg: PointStamped) -> Destination:
        with self.state_lock:
            key, index = self.next_clicked_destination_key()
            label_prefix = self.clicked_destination_label_prefix or "Clicked"
            destination = Destination(
                key=key,
                label=f"{label_prefix} {index:03d}",
                point=Point(
                    float(msg.point.x),
                    float(msg.point.y),
                    float(self.clicked_destination_z),
                ),
            )
            self.destinations.append(destination)

        if self.autosave_destinations:
            self.save_destinations()
        self.publish_markers()
        rospy.loginfo(
            "Added destination candidate from clicked point: %s (%s) -> [%.3f, %.3f, %.3f]",
            destination.label,
            destination.key,
            destination.point.x,
            destination.point.y,
            destination.point.z,
        )
        return destination

    def occupancy_callback(self, msg: OccupancyGrid) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            self.frame_id = msg.header.frame_id

    def path_callback(self, msg: RosPath) -> None:
        points = [
            Point(
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            )
            for pose in msg.poses
        ]
        with self.state_lock:
            self.latest_path_points = points
            self.latest_path_version += 1

    @staticmethod
    def point_distance(first: Point, second: Point) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)

    @staticmethod
    def copy_point(point: Point) -> Point:
        return Point(float(point.x), float(point.y), float(point.z))

    @staticmethod
    def path_length(points: list[Point]) -> float:
        return sum(
            math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y)
            for index in range(1, len(points))
        )

    def nearest_path_distance(self, points: list[Point], point: Point) -> float:
        if len(points) < 2:
            return 0.0

        cumulative = 0.0
        best_distance = float("inf")
        best_along_path = 0.0
        for index in range(1, len(points)):
            start = points[index - 1]
            end = points[index]
            dx = end.x - start.x
            dy = end.y - start.y
            segment_length_sq = dx * dx + dy * dy
            if segment_length_sq <= 0.0:
                continue

            t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / segment_length_sq
            t = max(0.0, min(1.0, t))
            projection_x = start.x + t * dx
            projection_y = start.y + t * dy
            distance = math.hypot(point.x - projection_x, point.y - projection_y)
            segment_length = math.sqrt(segment_length_sq)
            if distance < best_distance:
                best_distance = distance
                best_along_path = cumulative + segment_length * t
            cumulative += segment_length

        return best_along_path

    def interpolate_path_point(self, points: list[Point], distance_along_path: float) -> Point | None:
        if not points:
            return None
        if len(points) == 1 or distance_along_path <= 0.0:
            return self.copy_point(points[0])

        traveled = 0.0
        for index in range(1, len(points)):
            start = points[index - 1]
            end = points[index]
            segment_length = math.hypot(end.x - start.x, end.y - start.y)
            if segment_length <= 0.0:
                continue
            if traveled + segment_length >= distance_along_path:
                ratio = (distance_along_path - traveled) / segment_length
                return Point(
                    start.x + (end.x - start.x) * ratio,
                    start.y + (end.y - start.y) * ratio,
                    start.z + (end.z - start.z) * ratio,
                )
            traveled += segment_length

        return self.copy_point(points[-1])

    def auto_advance_current_callback(self, _event: rospy.TimerEvent) -> None:
        with self.state_lock:
            if not self.auto_advance_current:
                return
            if self.current_pose_overrides_auto_advance and self.received_current_pose:
                return
            current = self.copy_point(self.current_position) if self.current_position else None
            selected = self.selected_destination
            points = [self.copy_point(point) for point in self.latest_path_points]
            path_version = self.latest_path_version

        if current is None or selected is None or len(points) < 2:
            return
        if path_version == self.last_auto_advance_path_version:
            return

        total_length = self.path_length(points)
        if total_length <= 0.0:
            return

        current_distance = self.nearest_path_distance(points, current)
        target_distance = min(total_length, current_distance + self.auto_advance_distance)
        target = self.interpolate_path_point(points, target_distance)
        if target is None:
            return
        if self.point_distance(current, target) < self.auto_advance_min_move:
            self.last_auto_advance_path_version = path_version
            return

        msg = PointStamped()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        msg.point = target
        self.update_current_position_from_point(msg, "auto_path_advance")
        self.last_auto_advance_path_version = path_version

    def current_pose_callback(self, msg: PoseStamped) -> None:
        z = (
            self.current_pose_ground_z
            if self.current_pose_project_to_ground
            else msg.pose.position.z
        )
        source = self.current_pose_topic
        if self.current_pose_project_to_ground:
            source = f"{source} projected_to_ground"
        with self.state_lock:
            self.current_position = Point(
                msg.pose.position.x,
                msg.pose.position.y,
                z,
            )
            self.current_position_source = source
            self.received_current_pose = True
        self.publish_current_position()
        self.publish_markers()
        rospy.loginfo_throttle(
            2.0,
            "Updated current position from HLOC pose: [%.3f, %.3f, %.3f]",
            msg.pose.position.x,
            msg.pose.position.y,
            z,
        )

    def update_current_position_from_point(
        self, msg: PointStamped, source: str
    ) -> None:
        with self.state_lock:
            self.current_position = Point(msg.point.x, msg.point.y, msg.point.z)
            self.current_position_source = source
        self.publish_current_position()
        self.publish_markers()
        rospy.loginfo(
            "Updated current position from %s: [%.3f, %.3f, %.3f]",
            source,
            msg.point.x,
            msg.point.y,
            msg.point.z,
        )

    def current_point_callback(self, msg: PointStamped) -> None:
        self.update_current_position_from_point(msg, self.current_point_topic)

    def clicked_point_callback(self, msg: PointStamped) -> None:
        if self.clicked_point_mode == "sequence":
            if not self.sequence_current_position_set:
                self.sequence_current_position_set = True
                self.update_current_position_from_point(msg, self.clicked_point_topic)
                rospy.loginfo("First clicked point saved as current position.")
                return

            destination = self.add_destination_from_clicked_point(msg)
            self.select_destination(destination.key, source=self.clicked_point_topic)
            return

        if self.clicked_point_mode == "destination":
            destination = self.add_destination_from_clicked_point(msg)
            self.select_destination(destination.key, source=self.clicked_point_topic)
            return

        if self.clicked_point_mode == "candidate":
            self.add_destination_from_clicked_point(msg)
            return

        self.update_current_position_from_point(msg, self.clicked_point_topic)

    def selection_callback(self, msg: String) -> None:
        self.select_destination(msg.data, source="topic")

    def select_destination(self, raw_choice: str, source: str) -> bool:
        destination = self.parse_destination_choice(raw_choice)
        if destination is None:
            rospy.logwarn("Unknown destination choice: %s", raw_choice)
            self.print_destinations()
            return False

        with self.state_lock:
            self.selected_destination = destination

        goal = PointStamped()
        goal.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        goal.point = destination.point
        self.publish_current_position()
        self.selected_goal_pub.publish(goal)
        self.selected_goal_name_pub.publish(String(data=destination.key))
        self.publish_markers()
        self.ensure_rviz_running()

        rospy.loginfo(
            "Selected destination from %s: %s (%s) -> [%.3f, %.3f, %.3f]",
            source,
            destination.label,
            destination.key,
            destination.point.x,
            destination.point.y,
            destination.point.z,
        )
        return True

    def parse_destination_choice(self, raw_choice: str) -> Destination | None:
        choice = raw_choice.strip()
        if not choice:
            return None

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(self.destinations):
                return self.destinations[index]
            return None

        choice_lower = choice.lower()
        for destination in self.destinations:
            if choice_lower in (
                destination.key.lower(),
                destination.label.lower(),
            ):
                return destination
        return None

    def prompt_loop(self) -> None:
        if not sys.stdin.isatty():
            rospy.logwarn(
                "Destination prompt disabled because stdin is not interactive."
            )
            return

        self.print_destinations()
        while not rospy.is_shutdown():
            try:
                choice = input(
                    "\nSelect destination by index/name "
                    "[list/status/quit]: "
                ).strip()
            except EOFError:
                rospy.logwarn("Destination prompt ended (stdin closed).")
                return

            if not choice:
                continue

            command = choice.lower()
            if command in {"list", "ls"}:
                self.print_destinations()
                continue
            if command in {"status", "st"}:
                self.print_status()
                continue
            if command in {"quit", "q", "exit"}:
                rospy.loginfo("Destination prompt exited by user.")
                return

            self.select_destination(choice, source="terminal")

    def print_destinations(self) -> None:
        print("\nAvailable destinations:")
        for index, destination in enumerate(self.destinations, start=1):
            print(
                f"  [{index}] {destination.label} ({destination.key})"
                f" -> [{destination.point.x:.3f}, {destination.point.y:.3f},"
                f" {destination.point.z:.3f}]"
            )
        print(
            f"Use RViz Publish Point on {self.current_point_topic} to set the current"
            " position until localization is connected."
        )

    def print_status(self) -> None:
        with self.state_lock:
            current = self.current_position
            selected = self.selected_destination
            current_source = self.current_position_source

        if current is None:
            current_text = "current position: not set"
        else:
            current_text = (
                "current position: "
                f"[{current.x:.3f}, {current.y:.3f}, {current.z:.3f}]"
                f" from {current_source}"
            )

        if selected is None:
            selected_text = "selected destination: none"
        else:
            selected_text = (
                f"selected destination: {selected.label} ({selected.key})"
                f" -> [{selected.point.x:.3f}, {selected.point.y:.3f},"
                f" {selected.point.z:.3f}]"
            )

        print(current_text)
        print(selected_text)

    @staticmethod
    def resolve_default_rviz_config_path() -> Path:
        package_path = Path(rospkg.RosPack().get_path("destination_selector"))
        return package_path / "rviz" / "destination_selector.rviz"

    def ensure_rviz_running(self) -> None:
        if not self.rviz_on_select:
            return
        if self.rviz_process is not None and self.rviz_process.poll() is None:
            return
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            rospy.logerr(
                "RViz was requested, but no graphical display is available."
            )
            return
        if not self.rviz_config_path.exists():
            rospy.logwarn(
                "RViz config not found, starting RViz with defaults: %s",
                self.rviz_config_path,
            )
            command = ["rviz", "-f", self.frame_id]
        else:
            command = [
                "rviz",
                "-d",
                str(self.rviz_config_path),
                "-f",
                self.frame_id,
            ]

        try:
            self.rviz_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        except FileNotFoundError:
            rospy.logerr("RViz was requested, but the rviz executable was not found.")
        except OSError as error:
            rospy.logerr("Failed to start RViz: %s", error)
        else:
            rospy.loginfo("Started RViz for destination visualization.")

    def run_gui(self) -> bool:
        if tk is None or ttk is None:
            rospy.logerr(
                "Destination GUI requested, but tkinter is not available."
            )
            return False

        try:
            root = tk.Tk()
        except TclError as error:
            rospy.logerr("Destination GUI could not start: %s", error)
            return False

        root.title(self.gui_title)
        root.minsize(420, 300)

        selected_key_var = tk.StringVar(value="")
        selected_text_var = tk.StringVar(value="Selected: none")
        current_text_var = tk.StringVar(value="Current: not set")
        status_text_var = tk.StringVar(value="Choose a destination.")

        frame = ttk.Frame(root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Destination").grid(row=0, column=0, sticky="w")

        listbox = tk.Listbox(
            frame,
            height=min(max(len(self.destinations), 5), 10),
            exportselection=False,
        )
        listbox.grid(row=1, column=0, sticky="nsew", pady=(6, 8))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(6, 8))
        listbox.configure(yscrollcommand=scrollbar.set)

        for index, destination in enumerate(self.destinations, start=1):
            listbox.insert(
                "end",
                (
                    f"{index}. {destination.label} ({destination.key})"
                    f"  [{destination.point.x:.2f},"
                    f" {destination.point.y:.2f}, {destination.point.z:.2f}]"
                ),
            )

        def sync_listbox() -> None:
            if listbox.size() == len(self.destinations):
                return
            selected_key = selected_key_var.get()
            listbox.delete(0, "end")
            for index, destination in enumerate(self.destinations, start=1):
                listbox.insert(
                    "end",
                    (
                        f"{index}. {destination.label} ({destination.key})"
                        f"  [{destination.point.x:.2f},"
                        f" {destination.point.y:.2f}, {destination.point.z:.2f}]"
                    ),
                )
                if destination.key == selected_key:
                    listbox.selection_set(index - 1)
                    listbox.see(index - 1)

        status_frame = ttk.Frame(frame)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        status_frame.columnconfigure(0, weight=1)

        ttk.Label(status_frame, textvariable=selected_text_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(status_frame, textvariable=current_text_var).grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )
        ttk.Label(status_frame, textvariable=status_text_var).grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def select_from_listbox() -> None:
            selected_indices = listbox.curselection()
            if not selected_indices:
                status_text_var.set("Choose a destination first.")
                return

            destination = self.destinations[int(selected_indices[0])]
            if self.select_destination(destination.key, source="gui"):
                selected_key_var.set(destination.key)
                status_text_var.set(f"Published goal: {destination.label}")
                refresh_status()

        def refresh_status() -> None:
            sync_listbox()
            with self.state_lock:
                current = self.current_position
                selected = self.selected_destination

            if selected is None:
                selected_text_var.set("Selected: none")
            else:
                selected_text_var.set(
                    f"Selected: {selected.label} ({selected.key})"
                )
                if selected_key_var.get() != selected.key:
                    selected_key_var.set(selected.key)
                    for index, destination in enumerate(self.destinations):
                        if destination.key == selected.key:
                            listbox.selection_clear(0, "end")
                            listbox.selection_set(index)
                            listbox.see(index)
                            break

            if current is None:
                current_text_var.set("Current: not set")
            else:
                current_text_var.set(
                    f"Current: [{current.x:.2f}, {current.y:.2f},"
                    f" {current.z:.2f}]"
                )

        def refresh_later() -> None:
            if rospy.is_shutdown():
                root.after(50, root.destroy)
                return
            refresh_status()
            root.after(max(self.gui_refresh_ms, 100), refresh_later)

        def close_window() -> None:
            if self.shutdown_on_gui_close and not rospy.is_shutdown():
                rospy.signal_shutdown("Destination selector GUI closed")
            root.destroy()

        ttk.Button(button_frame, text="Select", command=select_from_listbox).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(button_frame, text="Close", command=close_window).grid(
            row=0, column=1
        )

        listbox.bind("<Double-Button-1>", lambda _event: select_from_listbox())
        listbox.bind("<Return>", lambda _event: select_from_listbox())
        root.protocol("WM_DELETE_WINDOW", close_window)

        if self.destinations:
            listbox.selection_set(0)
            listbox.activate(0)

        refresh_status()
        marker_thread = threading.Thread(
            target=self.spin,
            name="destination_selector_marker_spin",
            daemon=True,
        )
        marker_thread.start()
        rospy.loginfo("Destination GUI started.")
        root.after(max(self.gui_refresh_ms, 100), refresh_later)
        root.mainloop()
        return True

    def make_marker(
        self,
        marker_id: int,
        namespace: str,
        marker_type: int,
        action: int = Marker.ADD,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = action
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0)
        return marker

    def make_delete_marker(self, marker_id: int, namespace: str) -> Marker:
        return self.make_marker(marker_id, namespace, Marker.SPHERE, Marker.DELETE)

    def add_candidate_delete_markers(self, marker_array: MarkerArray) -> None:
        delete_count = max(self.candidate_marker_delete_count, len(self.destinations))
        for index in range(delete_count):
            marker_array.markers.append(
                self.make_delete_marker(index, "candidate_points")
            )
            marker_array.markers.append(
                self.make_delete_marker(1000 + index, "candidate_labels")
            )

    def build_markers(self) -> MarkerArray:
        marker_array = MarkerArray()

        with self.state_lock:
            current = self.current_position
            selected = self.selected_destination
            destinations = list(self.destinations)

        self.add_candidate_delete_markers(marker_array)

        if self.show_candidate_markers:
            for index, destination in enumerate(destinations):
                candidate_marker = self.make_marker(
                    marker_id=index,
                    namespace="candidate_points",
                    marker_type=Marker.SPHERE,
                )
                candidate_marker.pose.position = Point(
                    destination.point.x,
                    destination.point.y,
                    destination.point.z + self.marker_z_offset,
                )
                candidate_marker.scale = Vector3(
                    self.candidate_marker_scale,
                    self.candidate_marker_scale,
                    self.candidate_marker_scale,
                )
                candidate_marker.color = ColorRGBA(1.0, 0.75, 0.05, 0.95)
                marker_array.markers.append(candidate_marker)

                candidate_text = self.make_marker(
                    marker_id=1000 + index,
                    namespace="candidate_labels",
                    marker_type=Marker.TEXT_VIEW_FACING,
                )
                candidate_text.pose.position = Point(
                    destination.point.x,
                    destination.point.y,
                    destination.point.z + self.text_z_offset,
                )
                candidate_text.scale.z = 0.16
                candidate_text.color = ColorRGBA(1.0, 0.85, 0.1, 0.98)
                candidate_text.text = f"{index + 1}. {destination.label}"
                marker_array.markers.append(candidate_text)

        if selected is not None:
            goal_marker = self.make_marker(
                marker_id=2000,
                namespace="selected_goal",
                marker_type=Marker.SPHERE,
            )
            goal_marker.pose.position = Point(
                selected.point.x,
                selected.point.y,
                selected.point.z + self.marker_z_offset + 0.1,
            )
            goal_marker.scale = Vector3(
                self.goal_marker_scale,
                self.goal_marker_scale,
                self.goal_marker_scale,
            )
            goal_marker.color = ColorRGBA(1.0, 0.2, 0.2, 0.95)
            marker_array.markers.append(goal_marker)

            goal_text = self.make_marker(
                marker_id=2001,
                namespace="selected_goal_label",
                marker_type=Marker.TEXT_VIEW_FACING,
            )
            goal_text.pose.position = Point(
                selected.point.x,
                selected.point.y,
                selected.point.z + self.text_z_offset + 0.18,
            )
            goal_text.scale.z = 0.2
            goal_text.color = ColorRGBA(1.0, 0.85, 0.1, 0.98)
            goal_text.text = f"GOAL: {selected.label}"
            marker_array.markers.append(goal_text)
        else:
            marker_array.markers.append(
                self.make_delete_marker(2000, "selected_goal")
            )
            marker_array.markers.append(
                self.make_delete_marker(2001, "selected_goal_label")
            )

        if current is not None:
            body_marker = self.make_marker(
                marker_id=2002,
                namespace="current_position",
                marker_type=Marker.CYLINDER,
            )
            body_marker.pose.position = Point(
                current.x,
                current.y,
                current.z + self.marker_z_offset + 0.12,
            )
            body_marker.scale = Vector3(
                self.current_marker_scale * 0.75,
                self.current_marker_scale * 0.75,
                self.current_marker_scale * 0.2,
            )
            body_marker.color = ColorRGBA(0.1, 0.45, 1.0, 0.98)
            marker_array.markers.append(body_marker)

            head_marker = self.make_marker(
                marker_id=2004,
                namespace="current_position",
                marker_type=Marker.SPHERE,
            )
            head_marker.pose.position = Point(
                current.x,
                current.y + self.current_marker_scale * 0.42,
                current.z + self.marker_z_offset + 0.2,
            )
            head_marker.scale = Vector3(
                self.current_marker_scale * 0.5,
                self.current_marker_scale * 0.5,
                self.current_marker_scale * 0.5,
            )
            head_marker.color = ColorRGBA(0.15, 0.65, 1.0, 0.98)
            marker_array.markers.append(head_marker)

            arms_marker = self.make_marker(
                marker_id=2005,
                namespace="current_position",
                marker_type=Marker.LINE_STRIP,
            )
            arms_marker.points = [
                Point(
                    current.x - self.current_marker_scale * 0.55,
                    current.y,
                    current.z + self.marker_z_offset + 0.22,
                ),
                Point(
                    current.x + self.current_marker_scale * 0.55,
                    current.y,
                    current.z + self.marker_z_offset + 0.22,
                ),
            ]
            arms_marker.scale.x = 0.04
            arms_marker.color = ColorRGBA(0.15, 0.65, 1.0, 0.98)
            marker_array.markers.append(arms_marker)

            current_text = self.make_marker(
                marker_id=2003,
                namespace="current_position_label",
                marker_type=Marker.TEXT_VIEW_FACING,
            )
            current_text.pose.position = Point(
                current.x,
                current.y,
                current.z + self.text_z_offset + 0.08,
            )
            current_text.scale.z = 0.18
            current_text.color = ColorRGBA(0.45, 0.75, 1.0, 0.98)
            current_text.text = "CURRENT"
            marker_array.markers.append(current_text)
        else:
            marker_array.markers.append(
                self.make_delete_marker(2002, "current_position")
            )
            marker_array.markers.append(
                self.make_delete_marker(2003, "current_position_label")
            )
            marker_array.markers.append(
                self.make_delete_marker(2004, "current_position")
            )
            marker_array.markers.append(
                self.make_delete_marker(2005, "current_position")
            )

        return marker_array

    def publish_markers(self) -> None:
        self.marker_pub.publish(self.build_markers())

    def publish_current_position(self) -> None:
        with self.state_lock:
            current = self.current_position

        if current is None:
            return

        msg = PointStamped()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        msg.point = current
        self.current_position_pub.publish(msg)

    def spin(self) -> None:
        rate = rospy.Rate(max(self.publish_rate, 0.5))
        while not rospy.is_shutdown():
            self.publish_markers()
            self.publish_current_position()
            rate.sleep()


def main() -> None:
    rospy.init_node("destination_selector")
    node = DestinationSelectorNode()
    if node.gui_on_start:
        node.run_gui()
        if rospy.is_shutdown():
            return
    node.spin()


if __name__ == "__main__":
    main()
