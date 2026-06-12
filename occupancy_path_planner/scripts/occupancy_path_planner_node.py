#!/usr/bin/env python3

from __future__ import annotations

import heapq
import hashlib
import math
import threading
from collections import deque
from dataclasses import dataclass

import rospy
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker


@dataclass(frozen=True)
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    yaw: float
    cos_yaw: float
    sin_yaw: float


class OccupancyPathPlannerNode:
    def __init__(self) -> None:
        self.occupancy_topic = rospy.get_param(
            "~occupancy_topic", "/voxel_map/xy_occupancy"
        )
        self.start_topic = rospy.get_param(
            "~start_topic", "/destination_selector/current_position"
        )
        self.goal_topic = rospy.get_param(
            "~goal_topic", "/destination_selector/selected_goal"
        )
        self.path_topic = rospy.get_param("~path_topic", "/path_planner/path")
        self.path_marker_topic = rospy.get_param(
            "~path_marker_topic", "/path_planner/path_marker"
        )

        self.obstacle_threshold = int(rospy.get_param("~obstacle_threshold", 50))
        self.allow_unknown = self.get_bool_param("~allow_unknown", False)
        self.connectivity = int(rospy.get_param("~connectivity", 8))
        if self.connectivity not in (4, 8):
            rospy.logwarn("Unsupported connectivity %s; using 8.", self.connectivity)
            self.connectivity = 8
        self.inflation_radius = float(rospy.get_param("~inflation_radius", 0.0))
        self.snap_to_free_radius = float(
            rospy.get_param("~snap_to_free_radius", 0.6)
        )
        self.path_z_offset = float(rospy.get_param("~path_z_offset", 0.18))
        self.path_line_width = float(rospy.get_param("~path_line_width", 0.07))
        self.path_marker_alpha = float(rospy.get_param("~path_marker_alpha", 0.65))
        self.preferred_clearance = float(
            rospy.get_param("~preferred_clearance", 0.9)
        )
        self.clearance_cost_weight = float(
            rospy.get_param("~clearance_cost_weight", 4.0)
        )
        self.clearance_cost_power = float(
            rospy.get_param("~clearance_cost_power", 2.0)
        )
        self.centerline_cost_weight = float(
            rospy.get_param("~centerline_cost_weight", 12.0)
        )
        self.centerline_clearance = float(
            rospy.get_param("~centerline_clearance", 1.8)
        )
        self.centerline_cost_power = float(
            rospy.get_param("~centerline_cost_power", 2.0)
        )
        self.turn_cost_weight = float(rospy.get_param("~turn_cost_weight", 1.2))
        self.simplify_path = self.get_bool_param("~simplify_path", True)
        self.simplify_clearance_cost_margin = float(
            rospy.get_param("~simplify_clearance_cost_margin", 0.30)
        )
        self.replan_distance = float(rospy.get_param("~replan_distance", 1.0))

        self.state_lock = threading.Lock()
        self.map_msg: OccupancyGrid | None = None
        self.map_signature: str | None = None
        self.grid_info: GridInfo | None = None
        self.base_blocked: list[bool] = []
        self.blocked: list[bool] = []
        self.clearance_costs: list[float] = []
        self.clearance_distances: list[float] = []
        self.start_point: PointStamped | None = None
        self.goal_point: PointStamped | None = None
        self.last_path_points: list[Point] = []
        self.last_path_goal: Point | None = None

        self.path_pub = rospy.Publisher(
            self.path_topic, Path, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            self.path_marker_topic, Marker, queue_size=1, latch=True
        )

        self.map_sub = rospy.Subscriber(
            self.occupancy_topic,
            OccupancyGrid,
            self.map_callback,
            queue_size=1,
        )
        self.start_sub = rospy.Subscriber(
            self.start_topic,
            PointStamped,
            self.start_callback,
            queue_size=1,
        )
        self.goal_sub = rospy.Subscriber(
            self.goal_topic,
            PointStamped,
            self.goal_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "Occupancy path planner waiting for map=%s, start=%s, goal=%s",
            self.occupancy_topic,
            self.start_topic,
            self.goal_topic,
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
    def yaw_from_quaternion(quaternion: Quaternion) -> float:
        siny_cosp = 2.0 * (
            quaternion.w * quaternion.z + quaternion.x * quaternion.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        )
        return math.atan2(siny_cosp, cosy_cosp)

    def map_callback(self, msg: OccupancyGrid) -> None:
        map_signature = self.make_map_signature(msg)
        with self.state_lock:
            if self.map_signature == map_signature and self.map_msg is not None:
                return

        grid_info = self.make_grid_info(msg)
        base_blocked, blocked = self.make_blocked_grid(msg, grid_info)
        clearance_costs, clearance_distances = self.make_clearance_costs(
            base_blocked,
            grid_info,
        )
        with self.state_lock:
            self.map_msg = msg
            self.map_signature = map_signature
            self.grid_info = grid_info
            self.base_blocked = base_blocked
            self.blocked = blocked
            self.clearance_costs = clearance_costs
            self.clearance_distances = clearance_distances
        self.try_plan(force_replan=True, reason="map_changed")

    def start_callback(self, msg: PointStamped) -> None:
        with self.state_lock:
            self.start_point = msg
        self.try_plan(force_replan=False, reason="start_update")

    def goal_callback(self, msg: PointStamped) -> None:
        with self.state_lock:
            self.goal_point = msg
        self.try_plan(force_replan=True, reason="goal_changed")

    def make_map_signature(self, msg: OccupancyGrid) -> str:
        digest = hashlib.blake2b(digest_size=16)
        info = msg.info
        metadata = (
            int(info.width),
            int(info.height),
            float(info.resolution),
            float(info.origin.position.x),
            float(info.origin.position.y),
            float(info.origin.position.z),
            float(info.origin.orientation.x),
            float(info.origin.orientation.y),
            float(info.origin.orientation.z),
            float(info.origin.orientation.w),
        )
        digest.update(repr(metadata).encode("utf-8"))
        digest.update(bytes((int(value) + 256) % 256 for value in msg.data))
        return digest.hexdigest()

    def make_grid_info(self, msg: OccupancyGrid) -> GridInfo:
        yaw = self.yaw_from_quaternion(msg.info.origin.orientation)
        return GridInfo(
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
            yaw=yaw,
            cos_yaw=math.cos(yaw),
            sin_yaw=math.sin(yaw),
        )

    def make_blocked_grid(
        self, msg: OccupancyGrid, grid_info: GridInfo
    ) -> tuple[list[bool], list[bool]]:
        base_blocked = [
            self.is_occupied_value(value) for value in list(msg.data)
        ]
        if self.inflation_radius <= 0.0:
            return base_blocked, base_blocked[:]

        radius_cells = int(math.ceil(self.inflation_radius / grid_info.resolution))
        if radius_cells <= 0:
            return base_blocked, base_blocked[:]

        inflated = base_blocked[:]
        offsets: list[tuple[int, int]] = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if math.hypot(dx, dy) <= radius_cells:
                    offsets.append((dx, dy))

        for y in range(grid_info.height):
            for x in range(grid_info.width):
                if not base_blocked[self.index(x, y, grid_info)]:
                    continue
                for dx, dy in offsets:
                    nx = x + dx
                    ny = y + dy
                    if self.in_bounds(nx, ny, grid_info):
                        inflated[self.index(nx, ny, grid_info)] = True

        return base_blocked, inflated

    def make_clearance_costs(
        self, base_blocked: list[bool], grid_info: GridInfo
    ) -> tuple[list[float], list[float]]:
        use_preferred = (
            self.clearance_cost_weight > 0.0 and self.preferred_clearance > 0.0
        )
        use_centerline = (
            self.centerline_cost_weight > 0.0 and self.centerline_clearance > 0.0
        )
        if not use_preferred and not use_centerline:
            return [0.0] * len(base_blocked), [0.0] * len(base_blocked)

        preferred_cells = self.preferred_clearance / grid_info.resolution
        centerline_cells = self.centerline_clearance / grid_info.resolution
        max_distance = max(
            1.0,
            preferred_cells if use_preferred else 0.0,
            centerline_cells if use_centerline else 0.0,
        )
        distances = [float("inf")] * len(base_blocked)
        open_heap: list[tuple[float, int, int]] = []

        for y in range(grid_info.height):
            row_index = y * grid_info.width
            for x in range(grid_info.width):
                index = row_index + x
                if base_blocked[index]:
                    distances[index] = 0.0
                    heapq.heappush(open_heap, (0.0, x, y))

        if not open_heap:
            return [0.0] * len(base_blocked), [0.0] * len(base_blocked)

        steps = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        ]

        while open_heap and not rospy.is_shutdown():
            distance, x, y = heapq.heappop(open_heap)
            index = self.index(x, y, grid_info)
            if distance > distances[index] or distance >= max_distance:
                continue

            for dx, dy, step_cost in steps:
                nx = x + dx
                ny = y + dy
                if not self.in_bounds(nx, ny, grid_info):
                    continue
                next_distance = distance + step_cost
                if next_distance >= max_distance:
                    continue
                neighbor_index = self.index(nx, ny, grid_info)
                if next_distance >= distances[neighbor_index]:
                    continue
                distances[neighbor_index] = next_distance
                heapq.heappush(open_heap, (next_distance, nx, ny))

        costs: list[float] = []
        clearance_power = max(self.clearance_cost_power, 0.1)
        centerline_power = max(self.centerline_cost_power, 0.1)
        for blocked, distance in zip(base_blocked, distances):
            if blocked:
                costs.append(0.0)
                continue

            cell_cost = 0.0
            if use_preferred and distance < preferred_cells:
                deficit = (preferred_cells - distance) / preferred_cells
                cell_cost += self.clearance_cost_weight * (deficit**clearance_power)
            if use_centerline and distance < centerline_cells:
                deficit = (centerline_cells - distance) / centerline_cells
                cell_cost += self.centerline_cost_weight * (deficit**centerline_power)
            costs.append(cell_cost)
        return costs, distances

    def is_occupied_value(self, value: int) -> bool:
        if value < 0:
            return not self.allow_unknown
        return value >= self.obstacle_threshold

    @staticmethod
    def index(x: int, y: int, grid_info: GridInfo) -> int:
        return y * grid_info.width + x

    @staticmethod
    def in_bounds(x: int, y: int, grid_info: GridInfo) -> bool:
        return 0 <= x < grid_info.width and 0 <= y < grid_info.height

    def is_free(self, x: int, y: int, grid_info: GridInfo, blocked: list[bool]) -> bool:
        return self.in_bounds(x, y, grid_info) and not blocked[
            self.index(x, y, grid_info)
        ]

    def world_to_grid(self, point: Point, grid_info: GridInfo) -> tuple[int, int]:
        dx = point.x - grid_info.origin_x
        dy = point.y - grid_info.origin_y
        local_x = grid_info.cos_yaw * dx + grid_info.sin_yaw * dy
        local_y = -grid_info.sin_yaw * dx + grid_info.cos_yaw * dy
        return (
            int(math.floor(local_x / grid_info.resolution)),
            int(math.floor(local_y / grid_info.resolution)),
        )

    def grid_to_world(self, cell: tuple[int, int], grid_info: GridInfo) -> Point:
        local_x = (cell[0] + 0.5) * grid_info.resolution
        local_y = (cell[1] + 0.5) * grid_info.resolution
        return Point(
            grid_info.origin_x
            + grid_info.cos_yaw * local_x
            - grid_info.sin_yaw * local_y,
            grid_info.origin_y
            + grid_info.sin_yaw * local_x
            + grid_info.cos_yaw * local_y,
            self.path_z_offset,
        )

    def nearest_free_cell(
        self,
        cell: tuple[int, int],
        grid_info: GridInfo,
        blocked: list[bool],
    ) -> tuple[int, int] | None:
        search_radius = max(
            self.snap_to_free_radius,
            self.inflation_radius + grid_info.resolution,
        )
        max_radius_cells = int(math.ceil(search_radius / grid_info.resolution))
        start = (
            min(max(cell[0], 0), grid_info.width - 1),
            min(max(cell[1], 0), grid_info.height - 1),
        )

        if self.is_free(start[0], start[1], grid_info, blocked):
            return start

        queue: deque[tuple[int, int]] = deque([start])
        visited = {start}
        best: tuple[int, int] | None = None
        best_distance = float("inf")

        while queue:
            current = queue.popleft()
            radius = max(abs(current[0] - start[0]), abs(current[1] - start[1]))
            if radius > max_radius_cells:
                continue

            if self.is_free(current[0], current[1], grid_info, blocked):
                distance = math.hypot(current[0] - cell[0], current[1] - cell[1])
                if distance < best_distance:
                    best = current
                    best_distance = distance

            if best is not None and radius >= int(math.ceil(best_distance)):
                break

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in visited:
                    continue
                if not self.in_bounds(neighbor[0], neighbor[1], grid_info):
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        return best

    def endpoint_connector(
        self,
        cell: tuple[int, int],
        grid_info: GridInfo,
        base_blocked: list[bool],
        inflated_blocked: list[bool],
    ) -> list[tuple[int, int]] | None:
        start = (
            min(max(cell[0], 0), grid_info.width - 1),
            min(max(cell[1], 0), grid_info.height - 1),
        )
        if not self.is_free(start[0], start[1], grid_info, base_blocked):
            nearest = self.nearest_free_cell(start, grid_info, base_blocked)
            if nearest is None:
                return None
            start = nearest

        if self.is_free(start[0], start[1], grid_info, inflated_blocked):
            return [start]

        connector_radius = max(
            self.snap_to_free_radius,
            self.inflation_radius + grid_info.resolution,
        )
        max_radius_cells = int(math.ceil(connector_radius / grid_info.resolution))
        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far = {start: 0.0}

        while open_heap and not rospy.is_shutdown():
            current_cost, current = heapq.heappop(open_heap)
            if current_cost > cost_so_far.get(current, float("inf")):
                continue

            if self.is_free(current[0], current[1], grid_info, inflated_blocked):
                return self.reconstruct_path(came_from, current)

            for neighbor, step_cost in self.neighbors(current, grid_info, base_blocked):
                distance_from_start = math.hypot(
                    neighbor[0] - start[0],
                    neighbor[1] - start[1],
                )
                if distance_from_start > max_radius_cells:
                    continue
                next_cost = current_cost + step_cost
                if next_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                cost_so_far[neighbor] = next_cost
                heapq.heappush(open_heap, (next_cost, neighbor))

        return None

    @staticmethod
    def combine_endpoint_paths(
        start_connector: list[tuple[int, int]],
        main_path: list[tuple[int, int]],
        goal_connector: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        combined = start_connector[:]
        combined.extend(main_path[1:])
        combined.extend(reversed(goal_connector[:-1]))
        return combined

    def exact_path_point(self, point: Point) -> Point:
        return Point(point.x, point.y, self.path_z_offset)

    def try_plan(self, force_replan: bool, reason: str) -> None:
        with self.state_lock:
            map_msg = self.map_msg
            grid_info = self.grid_info
            base_blocked = self.base_blocked[:]
            blocked = self.blocked[:]
            clearance_costs = self.clearance_costs[:]
            start_point = self.start_point
            goal_point = self.goal_point
            last_path_points = self.last_path_points[:]
            last_path_goal = self.copy_point(self.last_path_goal)

        if (
            map_msg is None
            or grid_info is None
            or not base_blocked
            or start_point is None
            or goal_point is None
        ):
            return

        if not force_replan and self.should_keep_existing_path(
            start_point.point,
            goal_point.point,
            last_path_points,
            last_path_goal,
        ):
            rospy.logdebug_throttle(
                2.0,
                "Keeping current path; start is within %.2fm of it.",
                self.replan_distance,
            )
            trimmed = self.trim_path_from_start(start_point.point, last_path_points)
            if trimmed is not None:
                self.publish_path(trimmed, map_msg.header.frame_id)
            return

        start_connector = self.endpoint_connector(
            self.world_to_grid(start_point.point, grid_info),
            grid_info,
            base_blocked,
            blocked,
        )
        goal_connector = self.endpoint_connector(
            self.world_to_grid(goal_point.point, grid_info),
            grid_info,
            base_blocked,
            blocked,
        )

        if start_connector is None or goal_connector is None:
            rospy.logwarn_throttle(
                5.0,
                "Could not plan: start or goal cannot connect to nearby clearance."
            )
            self.publish_empty_path(map_msg.header.frame_id)
            self.clear_last_path()
            return

        start_cell = start_connector[-1]
        goal_cell = goal_connector[-1]
        cells = self.find_path(
            start_cell,
            goal_cell,
            grid_info,
            blocked,
            clearance_costs,
        )
        if not cells:
            rospy.logwarn_throttle(
                5.0,
                "No free-space path found from %s to %s",
                start_cell,
                goal_cell,
            )
            self.publish_empty_path(map_msg.header.frame_id)
            self.clear_last_path()
            return

        cells = self.combine_endpoint_paths(start_connector, cells, goal_connector)
        raw_cell_count = len(cells)
        if self.simplify_path:
            cells = self.simplify_cell_path(
                cells,
                grid_info,
                blocked,
                clearance_costs,
            )
        points = [self.grid_to_world(cell, grid_info) for cell in cells]
        points[0] = self.exact_path_point(start_point.point)
        points[-1] = self.exact_path_point(goal_point.point)
        self.store_last_path(points, goal_point.point)
        self.publish_path(points, map_msg.header.frame_id)
        rospy.loginfo_throttle(
            5.0,
            "Published path with %d points from %s to %s; raw_cells=%d reason=%s",
            len(points),
            start_cell,
            goal_cell,
            raw_cell_count,
            reason,
        )

    def should_keep_existing_path(
        self,
        start: Point,
        goal: Point,
        path_points: list[Point],
        path_goal: Point | None,
    ) -> bool:
        if self.replan_distance <= 0.0 or len(path_points) < 2 or path_goal is None:
            return False
        if self.distance_xy(goal, path_goal) > max(0.05, self.replan_distance * 0.25):
            return False
        return self.distance_to_path_xy(start, path_points) < self.replan_distance

    @staticmethod
    def distance_xy(a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def distance_to_path_xy(self, point: Point, path_points: list[Point]) -> float:
        best = float("inf")
        for start, end in zip(path_points, path_points[1:]):
            best = min(best, self.distance_to_segment_xy(point, start, end))
        return best

    @staticmethod
    def distance_to_segment_xy(point: Point, start: Point, end: Point) -> float:
        vx = end.x - start.x
        vy = end.y - start.y
        wx = point.x - start.x
        wy = point.y - start.y
        length_sq = vx * vx + vy * vy
        if length_sq <= 1e-12:
            return math.hypot(point.x - start.x, point.y - start.y)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
        closest_x = start.x + t * vx
        closest_y = start.y + t * vy
        return math.hypot(point.x - closest_x, point.y - closest_y)

    def trim_path_from_start(
        self, start: Point, path_points: list[Point]
    ) -> list[Point] | None:
        """Return path_points trimmed to begin at the closest point to start.

        Returns None if the trimmed path would be less than two points (robot
        is already past the end of the path).
        """
        if len(path_points) < 2:
            return None

        best_dist = float("inf")
        best_seg = 0
        best_t = 0.0

        for i, (seg_start, seg_end) in enumerate(zip(path_points, path_points[1:])):
            vx = seg_end.x - seg_start.x
            vy = seg_end.y - seg_start.y
            wx = start.x - seg_start.x
            wy = start.y - seg_start.y
            length_sq = vx * vx + vy * vy
            if length_sq <= 1e-12:
                t = 0.0
            else:
                t = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
            closest_x = seg_start.x + t * vx
            closest_y = seg_start.y + t * vy
            dist = math.hypot(start.x - closest_x, start.y - closest_y)
            if dist < best_dist:
                best_dist = dist
                best_seg = i
                best_t = t

        trimmed = [self.exact_path_point(start)]
        trimmed.extend(path_points[best_seg + 1:])

        if len(trimmed) < 2:
            return None
        return trimmed

    def store_last_path(self, points: list[Point], goal: Point) -> None:
        with self.state_lock:
            self.last_path_points = [self.copy_point(point) for point in points]
            self.last_path_goal = self.copy_point(goal)

    def clear_last_path(self) -> None:
        with self.state_lock:
            self.last_path_points = []
            self.last_path_goal = None

    @staticmethod
    def copy_point(point: Point | None) -> Point | None:
        if point is None:
            return None
        return Point(point.x, point.y, point.z)

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        grid_info: GridInfo,
        blocked: list[bool],
        clearance_costs: list[float],
    ) -> list[tuple[int, int]]:
        if start == goal:
            return [start]

        open_heap: list[
            tuple[float, float, int, tuple[int, int], tuple[int, int] | None]
        ] = []
        push_order = 0
        heapq.heappush(
            open_heap,
            (self.heuristic(start, goal), 0.0, push_order, start, None),
        )
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        best_direction: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while open_heap and not rospy.is_shutdown():
            _priority, current_cost, _order, current, incoming_direction = heapq.heappop(open_heap)
            if current_cost > g_score.get(current, float("inf")):
                continue
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor, step_cost in self.neighbors(current, grid_info, blocked):
                outgoing_direction = (
                    neighbor[0] - current[0],
                    neighbor[1] - current[1],
                )
                next_cost = (
                    current_cost
                    + step_cost
                    + self.turn_cost_from_direction(
                        incoming_direction,
                        outgoing_direction,
                    )
                    + self.clearance_cost(neighbor, grid_info, clearance_costs)
                )
                if next_cost >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = next_cost
                best_direction[neighbor] = outgoing_direction
                priority = next_cost + self.heuristic(neighbor, goal)
                push_order += 1
                heapq.heappush(
                    open_heap,
                    (priority, next_cost, push_order, neighbor, outgoing_direction),
                )

        return []

    def turn_cost(
        self,
        current: tuple[int, int],
        neighbor: tuple[int, int],
        previous: tuple[int, int] | None,
    ) -> float:
        if self.turn_cost_weight <= 0.0 or previous is None:
            return 0.0

        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (neighbor[0] - current[0], neighbor[1] - current[1])
        return self.turn_cost_from_direction(incoming, outgoing)

    def turn_cost_from_direction(
        self,
        incoming: tuple[int, int] | None,
        outgoing: tuple[int, int],
    ) -> float:
        if self.turn_cost_weight <= 0.0 or incoming is None:
            return 0.0
        if incoming == outgoing:
            return 0.0

        incoming_length = math.hypot(incoming[0], incoming[1])
        outgoing_length = math.hypot(outgoing[0], outgoing[1])
        if incoming_length <= 0.0 or outgoing_length <= 0.0:
            return 0.0

        dot = (
            incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        ) / (incoming_length * outgoing_length)
        angle = math.acos(max(-1.0, min(1.0, dot)))
        return self.turn_cost_weight * (angle / (math.pi * 0.25))

    def simplify_cell_path(
        self,
        cells: list[tuple[int, int]],
        grid_info: GridInfo,
        blocked: list[bool],
        clearance_costs: list[float],
    ) -> list[tuple[int, int]]:
        if len(cells) < 3:
            return cells

        simplified = [cells[0]]
        anchor_index = 0
        while anchor_index < len(cells) - 1 and not rospy.is_shutdown():
            target_index = len(cells) - 1
            while target_index > anchor_index + 1:
                if self.can_simplify_segment(
                    cells,
                    anchor_index,
                    target_index,
                    grid_info,
                    blocked,
                    clearance_costs,
                ):
                    break
                target_index -= 1

            simplified.append(cells[target_index])
            anchor_index = target_index

        return simplified

    def can_simplify_segment(
        self,
        cells: list[tuple[int, int]],
        start_index: int,
        end_index: int,
        grid_info: GridInfo,
        blocked: list[bool],
        clearance_costs: list[float],
    ) -> bool:
        start = cells[start_index]
        end = cells[end_index]
        line_cells = self.sample_line_cells(start, end)
        if any(not self.is_free(x, y, grid_info, blocked) for x, y in line_cells):
            return False

        if not clearance_costs:
            return True

        original_cells = cells[start_index : end_index + 1]
        original_average, original_peak = self.clearance_cost_summary(
            original_cells,
            grid_info,
            clearance_costs,
        )
        line_average, line_peak = self.clearance_cost_summary(
            line_cells,
            grid_info,
            clearance_costs,
        )
        margin = max(0.0, self.simplify_clearance_cost_margin)
        return (
            line_average <= original_average + margin
            and line_peak <= original_peak + margin * 2.0
        )

    def line_is_free(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        grid_info: GridInfo,
        blocked: list[bool],
    ) -> bool:
        for x, y in self.sample_line_cells(start, end):
            if not self.is_free(x, y, grid_info, blocked):
                return False
        return True

    def sample_line_cells(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy)) * 2
        if steps <= 0:
            return [start]

        samples: list[tuple[int, int]] = []
        last_cell: tuple[int, int] | None = None
        for step in range(steps + 1):
            ratio = step / float(steps)
            cell = (
                int(round(start[0] + dx * ratio)),
                int(round(start[1] + dy * ratio)),
            )
            if cell != last_cell:
                samples.append(cell)
                last_cell = cell
        return samples

    def clearance_cost_summary(
        self,
        cells: list[tuple[int, int]],
        grid_info: GridInfo,
        clearance_costs: list[float],
    ) -> tuple[float, float]:
        if not cells:
            return 0.0, 0.0

        values: list[float] = []
        for x, y in cells:
            if not self.in_bounds(x, y, grid_info):
                continue
            index = self.index(x, y, grid_info)
            if 0 <= index < len(clearance_costs):
                values.append(clearance_costs[index])

        if not values:
            return 0.0, 0.0
        return sum(values) / float(len(values)), max(values)

    def clearance_cost(
        self,
        cell: tuple[int, int],
        grid_info: GridInfo,
        clearance_costs: list[float],
    ) -> float:
        index = self.index(cell[0], cell[1], grid_info)
        if not 0 <= index < len(clearance_costs):
            return 0.0
        return clearance_costs[index]

    def neighbors(
        self,
        cell: tuple[int, int],
        grid_info: GridInfo,
        blocked: list[bool],
    ) -> list[tuple[tuple[int, int], float]]:
        x, y = cell
        steps = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
        ]
        if self.connectivity == 8:
            diagonal_cost = math.sqrt(2.0)
            steps.extend(
                [
                    (1, 1, diagonal_cost),
                    (1, -1, diagonal_cost),
                    (-1, 1, diagonal_cost),
                    (-1, -1, diagonal_cost),
                ]
            )

        neighbors: list[tuple[tuple[int, int], float]] = []
        for dx, dy, cost in steps:
            nx = x + dx
            ny = y + dy
            if not self.is_free(nx, ny, grid_info, blocked):
                continue
            if dx != 0 and dy != 0:
                if not self.is_free(x + dx, y, grid_info, blocked):
                    continue
                if not self.is_free(x, y + dy, grid_info, blocked):
                    continue
            neighbors.append(((nx, ny), cost))

        return neighbors

    def heuristic(self, cell: tuple[int, int], goal: tuple[int, int]) -> float:
        dx = abs(cell[0] - goal[0])
        dy = abs(cell[1] - goal[1])
        if self.connectivity == 8:
            diagonal = min(dx, dy)
            straight = max(dx, dy) - diagonal
            return math.sqrt(2.0) * diagonal + straight
        return float(dx + dy)

    @staticmethod
    def reconstruct_path(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def publish_empty_path(self, frame_id: str) -> None:
        header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
        path = Path()
        path.header = header
        self.path_pub.publish(path)

        marker = Marker()
        marker.header = header
        marker.ns = "planned_path"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)

    def publish_path(self, points: list[Point], frame_id: str) -> None:
        header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
        path = Path(header=header)
        for point in points:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position = point
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

        marker = Marker()
        marker.header = header
        marker.ns = "planned_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.path_line_width
        marker.color = ColorRGBA(0.0, 1.0, 0.2, self.path_marker_alpha)
        marker.points = points
        self.marker_pub.publish(marker)

    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    rospy.init_node("occupancy_path_planner")
    node = OccupancyPathPlannerNode()
    node.spin()


if __name__ == "__main__":
    main()
