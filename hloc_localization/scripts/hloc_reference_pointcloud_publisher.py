#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as point_cloud2
from geometry_msgs.msg import Point, PointStamped, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from rospkg import RosPack
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray


PLY_SCALAR_FORMATS = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


@dataclass(frozen=True)
class PlyProperty:
    name: str
    kind: str


@dataclass(frozen=True)
class PlyHeader:
    fmt: str
    vertex_count: int
    vertex_properties: tuple[PlyProperty, ...]
    data_offset: int


@dataclass(frozen=True)
class PointCloudData:
    xyz: np.ndarray
    rgb: Optional[np.ndarray]
    ids: np.ndarray


@dataclass(frozen=True)
class ExclusionSphere:
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class ManualEdits:
    spheres: list[ExclusionSphere]
    deleted_ids: set[int]


@dataclass(frozen=True)
class OccupancyProjection:
    resolution: float
    origin_xy: np.ndarray
    width: int
    height: int
    data: list[int]
    occupied_cells: int


@dataclass(frozen=True)
class OccupancyBrushEdit:
    action: str
    center_xy: np.ndarray
    radius: float
    shape: str


def default_reference_dir() -> Path:
    package_path = Path(RosPack().get_path("hloc_localization"))
    return (package_path.parents[1] / "hloc_reference").resolve()


def read_ply_header(path: Path) -> PlyHeader:
    with path.open("rb") as file:
        first_line = file.readline()
        if first_line != b"ply\n":
            raise ValueError(f"{path} is not a PLY file")

        fmt = ""
        vertex_count = 0
        vertex_properties: list[PlyProperty] = []
        current_element = ""

        while True:
            line_bytes = file.readline()
            if not line_bytes:
                raise ValueError(f"{path} ended before end_header")

            line = line_bytes.decode("ascii", errors="strict").strip()
            if line == "end_header":
                return PlyHeader(
                    fmt=fmt,
                    vertex_count=vertex_count,
                    vertex_properties=tuple(vertex_properties),
                    data_offset=file.tell(),
                )

            if not line or line.startswith("comment"):
                continue

            parts = line.split()
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:1] == ["element"]:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("PLY list properties are not supported for vertices")
                vertex_properties.append(PlyProperty(name=parts[2], kind=parts[1]))


def load_ascii_vertices(path: Path, header: PlyHeader) -> PointCloudData:
    rows: list[list[float]] = []
    names = [prop.name for prop in header.vertex_properties]
    try:
        x_idx = names.index("x")
        y_idx = names.index("y")
        z_idx = names.index("z")
    except ValueError as error:
        raise ValueError("PLY vertex properties must include x, y, z") from error

    color_indices = color_property_indices(names)
    with path.open("rb") as file:
        file.seek(header.data_offset)
        for _ in range(header.vertex_count):
            values = file.readline().decode("ascii").strip().split()
            if not values:
                continue
            rows.append([float(value) for value in values])

    data = np.asarray(rows, dtype=np.float64)
    xyz = data[:, [x_idx, y_idx, z_idx]].astype(np.float32)
    rgb = extract_rgb_from_columns(data, color_indices)
    return PointCloudData(xyz=xyz, rgb=rgb, ids=np.arange(len(xyz), dtype=np.int64))


def load_binary_vertices(path: Path, header: PlyHeader) -> PointCloudData:
    endian = "<" if header.fmt == "binary_little_endian" else ">"
    names = [prop.name for prop in header.vertex_properties]
    try:
        x_idx = names.index("x")
        y_idx = names.index("y")
        z_idx = names.index("z")
    except ValueError as error:
        raise ValueError("PLY vertex properties must include x, y, z") from error

    color_indices = color_property_indices(names)
    fmt_parts = []
    for prop in header.vertex_properties:
        if prop.kind not in PLY_SCALAR_FORMATS:
            raise ValueError(f"Unsupported PLY scalar type: {prop.kind}")
        fmt_parts.append(PLY_SCALAR_FORMATS[prop.kind][0])

    row_struct = struct.Struct(endian + "".join(fmt_parts))
    xyz = np.empty((header.vertex_count, 3), dtype=np.float32)
    rgb = np.empty(header.vertex_count, dtype=np.uint32) if color_indices else None

    with path.open("rb") as file:
        file.seek(header.data_offset)
        for row_idx in range(header.vertex_count):
            row = row_struct.unpack(file.read(row_struct.size))
            xyz[row_idx] = (row[x_idx], row[y_idx], row[z_idx])
            if rgb is not None:
                rgb[row_idx] = pack_rgb(
                    row[color_indices[0]],
                    row[color_indices[1]],
                    row[color_indices[2]],
                )

    return PointCloudData(
        xyz=xyz,
        rgb=rgb,
        ids=np.arange(header.vertex_count, dtype=np.int64),
    )


def color_property_indices(names: list[str]) -> Optional[tuple[int, int, int]]:
    if {"red", "green", "blue"}.issubset(names):
        return names.index("red"), names.index("green"), names.index("blue")
    if {"r", "g", "b"}.issubset(names):
        return names.index("r"), names.index("g"), names.index("b")
    return None


def extract_rgb_from_columns(
    data: np.ndarray,
    indices: Optional[tuple[int, int, int]],
) -> Optional[np.ndarray]:
    if indices is None:
        return None

    colors = np.clip(data[:, list(indices)], 0, 255).astype(np.uint32)
    return (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]


def pack_rgb(red: object, green: object, blue: object) -> np.uint32:
    r = int(max(0, min(255, round(float(red)))))
    g = int(max(0, min(255, round(float(green)))))
    b = int(max(0, min(255, round(float(blue)))))
    return np.uint32((r << 16) | (g << 8) | b)


def load_ply(path: Path) -> PointCloudData:
    header = read_ply_header(path)
    if header.vertex_count <= 0:
        raise ValueError(f"{path} contains no vertices")

    if header.fmt == "ascii":
        return load_ascii_vertices(path, header)
    if header.fmt in {"binary_little_endian", "binary_big_endian"}:
        return load_binary_vertices(path, header)
    raise ValueError(f"Unsupported PLY format: {header.fmt}")


def maybe_apply_calibration(
    points: np.ndarray,
    calibration_path: Path,
) -> np.ndarray:
    if not calibration_path.exists():
        rospy.logwarn("Calibration file not found, publishing raw HLoc points: %s", calibration_path)
        return points

    with calibration_path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    scale = float(calibration.get("scale", 1.0))
    rotation = np.asarray(
        calibration.get("rotation_matrix", np.eye(3)),
        dtype=np.float64,
    ).reshape(3, 3)
    translation = np.asarray(
        calibration.get("translation", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)

    transformed = scale * (points.astype(np.float64) @ rotation.T) + translation
    rospy.loginfo("Applied HLoc map calibration from %s", calibration_path)
    return transformed.astype(np.float32)


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))

    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)

    if dot < -0.999999:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / np.linalg.norm(axis)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * (skew @ skew)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / np.dot(cross, cross))


def maybe_align_floor(
    points: np.ndarray,
    min_percentile: float,
    max_percentile: float,
    min_points: int,
    normal_min_z: float,
) -> np.ndarray:
    if len(points) < max(3, min_points):
        rospy.logwarn("Skipping floor alignment because there are too few points: %d", len(points))
        return points

    low = float(np.percentile(points[:, 2], min_percentile))
    high = float(np.percentile(points[:, 2], max_percentile))
    floor_points = points[(points[:, 2] >= low) & (points[:, 2] <= high)].astype(np.float64)
    if len(floor_points) < max(3, min_points):
        rospy.logwarn(
            "Skipping floor alignment because floor candidate count is too small: %d",
            len(floor_points),
        )
        return points

    centroid = floor_points.mean(axis=0)
    _, _, vh = np.linalg.svd(floor_points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal

    if normal[2] < normal_min_z:
        rospy.logwarn(
            "Skipping floor alignment because fitted plane normal is suspicious: [%.3f %.3f %.3f]",
            normal[0],
            normal[1],
            normal[2],
        )
        return points

    rotation = rotation_from_vectors(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    aligned = (points.astype(np.float64) - centroid) @ rotation.T + centroid
    aligned_floor = (floor_points - centroid) @ rotation.T + centroid
    floor_z = float(np.median(aligned_floor[:, 2]))
    aligned[:, 2] -= floor_z

    tilt_degrees = math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))
    rospy.loginfo(
        "Auto floor alignment applied: candidates=%d z_window=[%.3f, %.3f] tilt=%.2fdeg z_shift=%.3f",
        len(floor_points),
        low,
        high,
        tilt_degrees,
        floor_z,
    )
    return aligned.astype(np.float32)


def maybe_downsample(
    cloud: PointCloudData,
    max_points: int,
) -> PointCloudData:
    if max_points <= 0 or len(cloud.xyz) <= max_points:
        return cloud

    indices = np.linspace(0, len(cloud.xyz) - 1, max_points, dtype=np.int64)
    rgb = cloud.rgb[indices] if cloud.rgb is not None else None
    return PointCloudData(xyz=cloud.xyz[indices], rgb=rgb, ids=cloud.ids[indices])


def filter_cloud_by_z(
    cloud: PointCloudData,
    min_z: Optional[float],
    max_z: Optional[float],
) -> PointCloudData:
    if min_z is None and max_z is None:
        return cloud

    mask = np.ones(len(cloud.xyz), dtype=bool)
    if min_z is not None:
        mask &= cloud.xyz[:, 2] > min_z
    if max_z is not None:
        mask &= cloud.xyz[:, 2] < max_z

    rgb = cloud.rgb[mask] if cloud.rgb is not None else None
    return PointCloudData(xyz=cloud.xyz[mask], rgb=rgb, ids=cloud.ids[mask])


def apply_manual_edits(
    cloud: PointCloudData,
    edits: ManualEdits,
) -> PointCloudData:
    if not edits.spheres and not edits.deleted_ids:
        return cloud

    mask = np.ones(len(cloud.xyz), dtype=bool)
    if edits.deleted_ids:
        mask &= ~np.isin(cloud.ids, list(edits.deleted_ids))

    points = cloud.xyz.astype(np.float64)
    for sphere in edits.spheres:
        diff = points - sphere.center.reshape(1, 3)
        distance_squared = np.einsum("ij,ij->i", diff, diff)
        mask &= distance_squared > sphere.radius * sphere.radius

    rgb = cloud.rgb[mask] if cloud.rgb is not None else None
    return PointCloudData(xyz=cloud.xyz[mask], rgb=rgb, ids=cloud.ids[mask])


def load_manual_edits(path: Path) -> ManualEdits:
    if not path.exists():
        return ManualEdits(spheres=[], deleted_ids=set())

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    items = raw.get("spheres", raw) if isinstance(raw, dict) else raw
    spheres: list[ExclusionSphere] = []
    for item in items:
        if "center" in item:
            center = item["center"]
        else:
            center = [item["x"], item["y"], item["z"]]
        spheres.append(
            ExclusionSphere(
                center=np.asarray(center, dtype=np.float64).reshape(3),
                radius=float(item["radius"]),
            )
        )
    deleted_ids = set()
    if isinstance(raw, dict):
        deleted_ids = {int(value) for value in raw.get("deleted_ids", [])}
    return ManualEdits(spheres=spheres, deleted_ids=deleted_ids)


def save_manual_edits(path: Path, edits: ManualEdits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spheres": [
            {
                "center": [float(value) for value in sphere.center],
                "radius": float(sphere.radius),
            }
            for sphere in edits.spheres
        ],
        "deleted_ids": sorted(int(value) for value in edits.deleted_ids),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def load_occupancy_brush_edits(path: Path) -> list[OccupancyBrushEdit]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    items = raw.get("edits", raw) if isinstance(raw, dict) else raw
    edits: list[OccupancyBrushEdit] = []
    for item in items:
        action = str(item.get("action", "free")).strip().lower()
        if action in {"erase", "clear"}:
            action = "free"
        elif action in {"add", "wall", "occupy"}:
            action = "occupied"
        if action not in {"free", "occupied"}:
            rospy.logwarn("Skipping unknown occupancy brush action: %s", action)
            continue

        if "center" in item:
            center = item["center"][:2]
        else:
            center = [item["x"], item["y"]]
        edits.append(
            OccupancyBrushEdit(
                action=action,
                center_xy=np.asarray(center, dtype=np.float64).reshape(2),
                radius=float(item["radius"]),
                shape=str(item.get("shape", "circle")).strip().lower(),
            )
        )
    return edits


def save_occupancy_brush_edits(path: Path, edits: list[OccupancyBrushEdit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "edits": [
            {
                "action": edit.action,
                "center": [float(value) for value in edit.center_xy],
                "radius": float(edit.radius),
                "shape": edit.shape,
            }
            for edit in edits
        ]
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def get_optional_float_param(name: str) -> Optional[float]:
    value = rospy.get_param(name, "")
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def filter_points_by_z(
    points: np.ndarray,
    min_z: Optional[float],
    max_z: Optional[float],
) -> np.ndarray:
    if min_z is None and max_z is None:
        return points

    mask = np.ones(len(points), dtype=bool)
    if min_z is not None:
        mask &= points[:, 2] >= min_z
    if max_z is not None:
        mask &= points[:, 2] <= max_z
    return points[mask]


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.copy()

    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(radius * 2 + 1):
        for dx in range(radius * 2 + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.copy()

    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask, dtype=bool)
    for dy in range(radius * 2 + 1):
        for dx in range(radius * 2 + 1):
            result &= padded[dy : dy + height, dx : dx + width]
    return result


def binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.copy()
    return binary_erode(binary_dilate(mask, radius), radius)


def remove_small_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    min_cells = max(1, int(min_cells))
    if min_cells <= 1:
        return mask.copy()

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        component: list[tuple[int, int]] = []
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True

        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(component) >= min_cells:
            for y, x in component:
                kept[y, x] = True

    return kept


def make_occupancy_projection(
    points: np.ndarray,
    resolution: float,
    padding: float,
    min_z: Optional[float],
    max_z: Optional[float],
    min_cell_points: int,
    wall_min_z_span: float,
    wall_z_bin_size: float,
    wall_min_z_bins: int,
    closing_radius_cells: int,
    min_component_cells: int,
    inflation_radius_cells: int,
) -> OccupancyProjection:
    if resolution <= 0.0:
        raise ValueError("grid_resolution must be positive")

    filtered_points = filter_points_by_z(points, min_z, max_z)
    if len(filtered_points) == 0:
        raise ValueError("No HLoc reference points remain after occupancy z filtering")

    xy = filtered_points[:, :2].astype(np.float64)
    origin_xy = xy.min(axis=0) - max(padding, 0.0)
    cell_xy = np.floor((xy - origin_xy) / resolution).astype(np.int64)
    width = int(cell_xy[:, 0].max()) + 1
    height = int(cell_xy[:, 1].max()) + 1

    linear_indices = cell_xy[:, 1] * width + cell_xy[:, 0]
    counts = np.bincount(linear_indices, minlength=width * height)
    min_z_by_cell = np.full(width * height, np.inf, dtype=np.float64)
    max_z_by_cell = np.full(width * height, -np.inf, dtype=np.float64)
    np.minimum.at(min_z_by_cell, linear_indices, filtered_points[:, 2])
    np.maximum.at(max_z_by_cell, linear_indices, filtered_points[:, 2])
    z_spans = max_z_by_cell - min_z_by_cell

    if wall_z_bin_size > 0.0:
        z_reference = min_z if min_z is not None else float(filtered_points[:, 2].min())
        z_bin_indices = np.floor(
            (filtered_points[:, 2].astype(np.float64) - z_reference) / wall_z_bin_size
        ).astype(np.int64)
        z_bin_indices = np.maximum(z_bin_indices, 0)
        z_bin_count = int(z_bin_indices.max()) + 1
        occupied_z_bins = np.unique(linear_indices * z_bin_count + z_bin_indices)
        z_bins_by_cell = np.bincount(
            occupied_z_bins // z_bin_count,
            minlength=width * height,
        )
    else:
        z_bins_by_cell = np.zeros(width * height, dtype=np.int64)

    occupied = counts >= max(1, min_cell_points)
    if wall_min_z_span > 0.0:
        occupied &= z_spans >= wall_min_z_span
    if wall_min_z_bins > 1 and wall_z_bin_size > 0.0:
        occupied &= z_bins_by_cell >= wall_min_z_bins
    if not np.any(occupied):
        raise ValueError(
            "No occupied cells remain after wall evidence filtering; "
            "try lowering wall_min_z_span, wall_min_z_bins, or min_cell_points."
        )

    occupied_grid = occupied.reshape(height, width)
    occupied_grid = binary_close(occupied_grid, closing_radius_cells)
    occupied_grid = remove_small_components(occupied_grid, min_component_cells)
    occupied_grid = binary_dilate(occupied_grid, inflation_radius_cells)
    occupied = occupied_grid.reshape(width * height)

    grid = np.zeros(width * height, dtype=np.int8)
    grid[occupied] = 100
    return OccupancyProjection(
        resolution=resolution,
        origin_xy=origin_xy,
        width=width,
        height=height,
        data=grid.tolist(),
        occupied_cells=int(np.count_nonzero(occupied)),
    )


def apply_occupancy_brush_edits(
    projection: OccupancyProjection,
    edits: list[OccupancyBrushEdit],
) -> OccupancyProjection:
    if not edits:
        return projection

    grid = np.asarray(projection.data, dtype=np.int8).reshape(
        projection.height,
        projection.width,
    )
    origin_x = float(projection.origin_xy[0])
    origin_y = float(projection.origin_xy[1])
    resolution = float(projection.resolution)

    for edit in edits:
        radius = max(float(edit.radius), resolution * 0.5)
        center_x = float(edit.center_xy[0])
        center_y = float(edit.center_xy[1])
        min_x = max(0, int(math.floor((center_x - radius - origin_x) / resolution)))
        max_x = min(
            projection.width - 1,
            int(math.floor((center_x + radius - origin_x) / resolution)),
        )
        min_y = max(0, int(math.floor((center_y - radius - origin_y) / resolution)))
        max_y = min(
            projection.height - 1,
            int(math.floor((center_y + radius - origin_y) / resolution)),
        )
        if min_x > max_x or min_y > max_y:
            continue

        patch = grid[min_y : max_y + 1, min_x : max_x + 1]
        if edit.shape == "square":
            brush = np.ones_like(patch, dtype=bool)
        else:
            xs = origin_x + (np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5) * resolution
            ys = origin_y + (np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5) * resolution
            dx = xs.reshape(1, -1) - center_x
            dy = ys.reshape(-1, 1) - center_y
            brush = (dx * dx + dy * dy) <= radius * radius
        patch[brush] = 100 if edit.action == "occupied" else 0

    return OccupancyProjection(
        resolution=projection.resolution,
        origin_xy=projection.origin_xy,
        width=projection.width,
        height=projection.height,
        data=grid.reshape(projection.width * projection.height).tolist(),
        occupied_cells=int(np.count_nonzero(grid)),
    )


def make_occupancy_grid(
    projection: OccupancyProjection,
    frame_id: str,
    stamp: rospy.Time,
) -> OccupancyGrid:
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = float(projection.resolution)
    msg.info.width = int(projection.width)
    msg.info.height = int(projection.height)
    msg.info.origin = Pose()
    msg.info.origin.position.x = float(projection.origin_xy[0])
    msg.info.origin.position.y = float(projection.origin_xy[1])
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    msg.data = projection.data
    return msg


def make_bounds_marker(points: np.ndarray, frame_id: str, stamp: rospy.Time) -> Marker:
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    corners = [
        (min_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], max_corner[1], max_corner[2]),
        (min_corner[0], max_corner[1], max_corner[2]),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = "hloc_reference_bounds"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.03
    marker.color = ColorRGBA(0.1, 0.65, 1.0, 0.9)
    marker.lifetime = rospy.Duration(0)

    for start_idx, end_idx in edges:
        marker.points.append(Point(*corners[start_idx]))
        marker.points.append(Point(*corners[end_idx]))
    return marker


def points_for_deleted_ids(cloud: PointCloudData, deleted_ids: set[int]) -> np.ndarray:
    if not deleted_ids:
        return np.empty((0, 3), dtype=np.float32)
    return cloud.xyz[np.isin(cloud.ids, list(deleted_ids))]


def make_eraser_markers(
    edits: ManualEdits,
    deleted_points: np.ndarray,
    frame_id: str,
    stamp: rospy.Time,
    deleted_point_marker_size: float,
) -> MarkerArray:
    marker_array = MarkerArray()
    clear_marker = Marker()
    clear_marker.header.frame_id = frame_id
    clear_marker.header.stamp = stamp
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)

    for idx, sphere in enumerate(edits.spheres):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "manual_eraser_spheres"
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(*sphere.center)
        marker.pose.orientation.w = 1.0
        marker.scale.x = sphere.radius * 2.0
        marker.scale.y = sphere.radius * 2.0
        marker.scale.z = sphere.radius * 2.0
        marker.color = ColorRGBA(1.0, 0.2, 0.05, 0.22)
        marker.lifetime = rospy.Duration(0)
        marker_array.markers.append(marker)

    for idx, point in enumerate(deleted_points):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "deleted_points"
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(*point)
        marker.pose.orientation.w = 1.0
        marker.scale.x = deleted_point_marker_size
        marker.scale.y = deleted_point_marker_size
        marker.scale.z = deleted_point_marker_size
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.95)
        marker.lifetime = rospy.Duration(0)
        marker_array.markers.append(marker)
    return marker_array


def make_point_rows(cloud: PointCloudData) -> Iterable[tuple]:
    if cloud.rgb is None:
        for x, y, z in cloud.xyz:
            yield float(x), float(y), float(z)
        return

    for (x, y, z), rgb in zip(cloud.xyz, cloud.rgb):
        yield float(x), float(y), float(z), int(rgb)


def make_pointcloud2(
    cloud: PointCloudData,
    frame_id: str,
    stamp: rospy.Time,
) -> PointCloud2:
    header = Header(frame_id=frame_id, stamp=stamp)
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    if cloud.rgb is not None:
        fields.append(PointField("rgb", 12, PointField.UINT32, 1))

    return point_cloud2.create_cloud(header, fields, list(make_point_rows(cloud)))


def get_bool_param(name: str, default: bool) -> bool:
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_ply_path(reference_dir: Path, ply_path_param: str) -> Path:
    if ply_path_param.strip():
        return Path(ply_path_param).expanduser().resolve()
    return reference_dir / "debug" / "sparse_points_raw.ply"


def main() -> None:
    rospy.init_node("hloc_reference_pointcloud_publisher")

    reference_dir = Path(
        str(rospy.get_param("~reference_dir", str(default_reference_dir())))
    ).expanduser().resolve()
    ply_path = resolve_ply_path(reference_dir, str(rospy.get_param("~ply_path", "")))
    calibration_path = Path(
        str(rospy.get_param("~calibration_file", str(reference_dir.parent / "calibration.json")))
    ).expanduser().resolve()

    topic = str(rospy.get_param("~topic", "/hloc_reference/points"))
    occupancy_topic = str(
        rospy.get_param("~occupancy_topic", "/hloc_reference/xy_occupancy")
    )
    bounds_topic = str(rospy.get_param("~bounds_topic", "/hloc_reference/bounds"))
    frame_id = str(rospy.get_param("~frame_id", "map"))
    rate_hz = max(float(rospy.get_param("~rate", 0.5)), 0.01)
    max_points = int(rospy.get_param("~max_points", 0))
    use_calibration = get_bool_param("~use_calibration", True)
    auto_floor_align = get_bool_param("~auto_floor_align", True)
    floor_fit_min_percentile = float(rospy.get_param("~floor_fit_min_percentile", 1.0))
    floor_fit_max_percentile = float(rospy.get_param("~floor_fit_max_percentile", 25.0))
    floor_fit_min_points = int(rospy.get_param("~floor_fit_min_points", 1000))
    floor_normal_min_z = float(rospy.get_param("~floor_normal_min_z", 0.5))
    publish_occupancy = get_bool_param("~publish_occupancy", True)
    publish_bounds = get_bool_param("~publish_bounds", True)
    publish_erasers = get_bool_param("~publish_erasers", False)
    grid_resolution = float(rospy.get_param("~grid_resolution", 0.10))
    grid_padding = float(rospy.get_param("~grid_padding", 0.50))
    grid_min_z = get_optional_float_param("~grid_min_z")
    grid_max_z = get_optional_float_param("~grid_max_z")
    min_cell_points = int(rospy.get_param("~min_cell_points", 1))
    wall_min_z_span = float(rospy.get_param("~wall_min_z_span", 0.20))
    wall_z_bin_size = float(rospy.get_param("~wall_z_bin_size", 0.25))
    wall_min_z_bins = int(rospy.get_param("~wall_min_z_bins", 2))
    occupancy_closing_radius_cells = int(
        rospy.get_param("~occupancy_closing_radius_cells", 4)
    )
    occupancy_min_component_cells = int(
        rospy.get_param("~occupancy_min_component_cells", 20)
    )
    occupancy_inflation_radius_cells = int(
        rospy.get_param("~occupancy_inflation_radius_cells", 2)
    )
    cloud_min_z = get_optional_float_param("~cloud_min_z")
    cloud_max_z = get_optional_float_param("~cloud_max_z")
    edit_mode = get_bool_param("~edit_mode", True)
    edit_target = str(rospy.get_param("~edit_target", "pointcloud")).strip().lower()
    clicked_point_topic = str(rospy.get_param("~clicked_point_topic", "/clicked_point"))
    erase_mode = str(rospy.get_param("~erase_mode", "nearest")).strip().lower()
    erase_radius = float(rospy.get_param("~erase_radius", 0.30))
    erase_nearest_max_distance = float(
        rospy.get_param("~erase_nearest_max_distance", 0.15)
    )
    deleted_point_marker_size = float(rospy.get_param("~deleted_point_marker_size", 0.08))
    autosave_erasers = get_bool_param("~autosave_erasers", True)
    eraser_file = Path(
        str(rospy.get_param("~eraser_file", str(reference_dir / "debug" / "manual_erasers.json")))
    ).expanduser().resolve()
    eraser_topic = str(rospy.get_param("~eraser_topic", "/hloc_reference/erasers"))
    occupancy_edit_mode = str(
        rospy.get_param("~occupancy_edit_mode", "free")
    ).strip().lower()
    occupancy_edit_radius = float(rospy.get_param("~occupancy_edit_radius", 0.30))
    occupancy_edit_shape = str(
        rospy.get_param("~occupancy_edit_shape", "square")
    ).strip().lower()
    autosave_occupancy_edits = get_bool_param("~autosave_occupancy_edits", True)
    occupancy_edit_file = Path(
        str(
            rospy.get_param(
                "~occupancy_edit_file",
                str(reference_dir / "debug" / "manual_occupancy_edits.json"),
            )
        )
    ).expanduser().resolve()

    if not ply_path.exists():
        raise FileNotFoundError(f"HLoc reference point cloud not found: {ply_path}")
    if erase_mode not in {"nearest", "sphere"}:
        raise ValueError("erase_mode must be 'nearest' or 'sphere'")
    if edit_target not in {"pointcloud", "occupancy"}:
        raise ValueError("edit_target must be 'pointcloud' or 'occupancy'")

    rospy.loginfo("Loading HLoc reference point cloud: %s", ply_path)
    cloud = load_ply(ply_path)
    if use_calibration:
        cloud = PointCloudData(
            xyz=maybe_apply_calibration(cloud.xyz, calibration_path),
            rgb=cloud.rgb,
            ids=cloud.ids,
        )
    if auto_floor_align:
        cloud = PointCloudData(
            xyz=maybe_align_floor(
                cloud.xyz,
                min_percentile=floor_fit_min_percentile,
                max_percentile=floor_fit_max_percentile,
                min_points=floor_fit_min_points,
                normal_min_z=floor_normal_min_z,
            ),
            rgb=cloud.rgb,
            ids=cloud.ids,
        )

    base_cloud = filter_cloud_by_z(cloud, cloud_min_z, cloud_max_z)
    if len(base_cloud.xyz) != len(cloud.xyz):
        rospy.loginfo(
            "Filtered HLoc cloud by z (%s, %s): %d -> %d points",
            "-inf" if cloud_min_z is None else f"{cloud_min_z:.3f}",
            "+inf" if cloud_max_z is None else f"{cloud_max_z:.3f}",
            len(cloud.xyz),
            len(base_cloud.xyz),
        )

    lock = threading.Lock()
    manual_edits = load_manual_edits(eraser_file)
    occupancy_edits = load_occupancy_brush_edits(occupancy_edit_file)
    undo_stack: list[tuple[str, int]] = []
    occupancy_undo_stack: list[int] = []
    filtered_cloud = apply_manual_edits(base_cloud, manual_edits)
    occupancy = None

    if manual_edits.spheres or manual_edits.deleted_ids:
        rospy.loginfo(
            "Loaded manual edits from %s: spheres=%d deleted_points=%d points=%d -> %d",
            eraser_file,
            len(manual_edits.spheres),
            len(manual_edits.deleted_ids),
            len(base_cloud.xyz),
            len(filtered_cloud.xyz),
        )
    if occupancy_edits:
        rospy.loginfo(
            "Loaded manual occupancy edits from %s: edits=%d",
            occupancy_edit_file,
            len(occupancy_edits),
        )

    def rebuild_filtered_outputs() -> None:
        nonlocal filtered_cloud, occupancy
        filtered_cloud = apply_manual_edits(base_cloud, manual_edits)
        occupancy = None
        if publish_occupancy:
            occupancy = make_occupancy_projection(
                filtered_cloud.xyz,
                resolution=grid_resolution,
                padding=grid_padding,
                min_z=grid_min_z,
                max_z=grid_max_z,
                min_cell_points=min_cell_points,
                wall_min_z_span=wall_min_z_span,
                wall_z_bin_size=wall_z_bin_size,
                wall_min_z_bins=wall_min_z_bins,
                closing_radius_cells=occupancy_closing_radius_cells,
                min_component_cells=occupancy_min_component_cells,
                inflation_radius_cells=occupancy_inflation_radius_cells,
            )
            if occupancy_edits:
                occupancy = apply_occupancy_brush_edits(occupancy, occupancy_edits)
            rospy.loginfo(
                "Built HLoc occupancy grid from edited cloud: points=%d size=%dx%d resolution=%.3f occupied_cells=%d manual_occupancy_edits=%d",
                len(filtered_cloud.xyz),
                occupancy.width,
                occupancy.height,
                occupancy.resolution,
                occupancy.occupied_cells,
                len(occupancy_edits),
            )

    rebuild_filtered_outputs()

    def save_erasers_if_needed() -> None:
        if autosave_erasers:
            save_manual_edits(eraser_file, manual_edits)

    def save_occupancy_edits_if_needed() -> None:
        if autosave_occupancy_edits:
            save_occupancy_brush_edits(occupancy_edit_file, occupancy_edits)

    def normalize_occupancy_edit_mode(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"free", "erase", "clear"}:
            return "free"
        if normalized in {"occupied", "occupy", "add", "wall"}:
            return "occupied"
        raise ValueError("occupancy_edit_mode must be 'free' or 'occupied'")

    def normalize_occupancy_edit_shape(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"square", "box", "rect", "rectangle"}:
            return "square"
        if normalized in {"circle", "round"}:
            return "circle"
        raise ValueError("occupancy_edit_shape must be 'square' or 'circle'")

    def clicked_point_callback(msg: PointStamped) -> None:
        center = np.asarray([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        with lock:
            if edit_target == "occupancy":
                try:
                    mode = normalize_occupancy_edit_mode(
                        str(rospy.get_param("~occupancy_edit_mode", occupancy_edit_mode))
                    )
                    shape = normalize_occupancy_edit_shape(
                        str(rospy.get_param("~occupancy_edit_shape", occupancy_edit_shape))
                    )
                except ValueError as error:
                    rospy.logwarn("%s", error)
                    return
                radius = float(rospy.get_param("~occupancy_edit_radius", occupancy_edit_radius))
                occupancy_edits.append(
                    OccupancyBrushEdit(
                        action=mode,
                        center_xy=center[:2].copy(),
                        radius=radius,
                        shape=shape,
                    )
                )
                occupancy_undo_stack.append(len(occupancy_edits) - 1)
                rebuild_filtered_outputs()
                save_occupancy_edits_if_needed()
                rospy.loginfo(
                    "Painted occupancy grid action=%s shape=%s center=[%.3f, %.3f] radius=%.3f edits=%d occupied_cells=%d",
                    mode,
                    shape,
                    center[0],
                    center[1],
                    radius,
                    len(occupancy_edits),
                    0 if occupancy is None else occupancy.occupied_cells,
                )
                return

            if erase_mode == "nearest":
                if len(filtered_cloud.xyz) == 0:
                    rospy.logwarn("No points remain, so nearest-point erase was ignored.")
                    return
                diff = filtered_cloud.xyz.astype(np.float64) - center.reshape(1, 3)
                distance_squared = np.einsum("ij,ij->i", diff, diff)
                nearest_idx = int(np.argmin(distance_squared))
                nearest_distance = float(math.sqrt(distance_squared[nearest_idx]))
                if nearest_distance > erase_nearest_max_distance:
                    rospy.logwarn(
                        "Nearest point is %.3fm away, above erase_nearest_max_distance=%.3f. Click closer to a point.",
                        nearest_distance,
                        erase_nearest_max_distance,
                    )
                    return
                point_id = int(filtered_cloud.ids[nearest_idx])
                manual_edits.deleted_ids.add(point_id)
                undo_stack.append(("point", point_id))
            else:
                manual_edits.spheres.append(ExclusionSphere(center=center, radius=erase_radius))
                undo_stack.append(("sphere", len(manual_edits.spheres) - 1))

            rebuild_filtered_outputs()
            save_erasers_if_needed()
            if erase_mode == "nearest":
                rospy.loginfo(
                    "Deleted nearest point id=%d distance=%.3fm. Remaining points=%d",
                    point_id,
                    nearest_distance,
                    len(filtered_cloud.xyz),
                )
            else:
                rospy.loginfo(
                    "Added eraser sphere %d at [%.3f, %.3f, %.3f], radius=%.3f. Remaining points=%d",
                    len(manual_edits.spheres),
                    center[0],
                    center[1],
                    center[2],
                    erase_radius,
                    len(filtered_cloud.xyz),
                )

    def undo_occupancy_callback(_request: Trigger) -> TriggerResponse:
        with lock:
            if occupancy_undo_stack:
                index = occupancy_undo_stack.pop()
                if 0 <= index < len(occupancy_edits):
                    occupancy_edits.pop(index)
            elif occupancy_edits:
                occupancy_edits.pop()
            else:
                return TriggerResponse(success=False, message="No manual occupancy edits to undo.")
            rebuild_filtered_outputs()
            save_occupancy_edits_if_needed()
            return TriggerResponse(
                success=True,
                message=f"Removed last manual occupancy edit. Remaining edits={len(occupancy_edits)}",
            )

    def clear_occupancy_callback(_request: Trigger) -> TriggerResponse:
        with lock:
            occupancy_edits.clear()
            occupancy_undo_stack.clear()
            rebuild_filtered_outputs()
            save_occupancy_edits_if_needed()
            return TriggerResponse(success=True, message="Cleared all manual occupancy edits.")

    def save_occupancy_callback(_request: Trigger) -> TriggerResponse:
        with lock:
            save_occupancy_brush_edits(occupancy_edit_file, occupancy_edits)
            return TriggerResponse(
                success=True,
                message=f"Saved manual occupancy edits to {occupancy_edit_file}",
            )

    def undo_callback(_request: Trigger) -> TriggerResponse:
        with lock:
            if undo_stack:
                action, value = undo_stack.pop()
                if action == "point":
                    manual_edits.deleted_ids.discard(value)
                elif action == "sphere" and manual_edits.spheres:
                    manual_edits.spheres.pop()
            elif manual_edits.deleted_ids:
                manual_edits.deleted_ids.pop()
            elif manual_edits.spheres:
                manual_edits.spheres.pop()
            else:
                return TriggerResponse(success=False, message="No manual edits to undo.")
            rebuild_filtered_outputs()
            save_erasers_if_needed()
            return TriggerResponse(
                success=True,
                message=(
                    "Removed last manual edit. "
                    f"Remaining: spheres={len(manual_edits.spheres)} "
                    f"deleted_points={len(manual_edits.deleted_ids)}"
                ),
            )

    def clear_callback(_request: Trigger) -> TriggerResponse:
        with lock:
            manual_edits.spheres.clear()
            manual_edits.deleted_ids.clear()
            undo_stack.clear()
            rebuild_filtered_outputs()
            save_erasers_if_needed()
            return TriggerResponse(success=True, message="Cleared all manual edits.")

    def save_callback(_request: Trigger) -> TriggerResponse:
        with lock:
            save_manual_edits(eraser_file, manual_edits)
            return TriggerResponse(success=True, message=f"Saved manual edits to {eraser_file}")

    publisher = rospy.Publisher(topic, PointCloud2, queue_size=1, latch=True)
    occupancy_pub = None
    if publish_occupancy:
        occupancy_pub = rospy.Publisher(
            occupancy_topic, OccupancyGrid, queue_size=1, latch=True
        )
    bounds_pub = None
    if publish_bounds:
        bounds_pub = rospy.Publisher(bounds_topic, Marker, queue_size=1, latch=True)
    eraser_pub = rospy.Publisher(eraser_topic, MarkerArray, queue_size=1, latch=True)
    if edit_mode:
        rospy.Subscriber(clicked_point_topic, PointStamped, clicked_point_callback, queue_size=10)
        rospy.Service("~undo_eraser", Trigger, undo_callback)
        rospy.Service("~clear_erasers", Trigger, clear_callback)
        rospy.Service("~save_erasers", Trigger, save_callback)
        rospy.Service("~undo_occupancy_edit", Trigger, undo_occupancy_callback)
        rospy.Service("~clear_occupancy_edits", Trigger, clear_occupancy_callback)
        rospy.Service("~save_occupancy_edits", Trigger, save_occupancy_callback)
        if edit_target == "occupancy":
            rospy.loginfo(
                "Manual occupancy edit mode is enabled: click RViz Publish Point on %s; mode=%s shape=%s radius=%.3f file=%s. Change live with rosparam set /hloc_reference_pointcloud_publisher/occupancy_edit_mode free|occupied and /hloc_reference_pointcloud_publisher/occupancy_edit_shape square|circle.",
                clicked_point_topic,
                occupancy_edit_mode,
                occupancy_edit_shape,
                occupancy_edit_radius,
                occupancy_edit_file,
            )
        else:
            rospy.loginfo(
                "Manual pointcloud edit mode is enabled: mode=%s click RViz Publish Point on %s; radius=%.3f nearest_max_distance=%.3f file=%s",
                erase_mode,
                clicked_point_topic,
                erase_radius,
                erase_nearest_max_distance,
                eraser_file,
            )
    rospy.sleep(0.5)

    rospy.loginfo(
        "Publishing HLoc reference cloud: points=%d topic=%s frame=%s",
        len(filtered_cloud.xyz),
        topic,
        frame_id,
    )
    if publish_occupancy:
        rospy.loginfo("Publishing HLoc occupancy grid: topic=%s", occupancy_topic)
    if publish_bounds:
        rospy.loginfo("Publishing HLoc reference bounds: topic=%s", bounds_topic)
    if publish_erasers:
        rospy.loginfo("Publishing manual edit markers: topic=%s", eraser_topic)
    else:
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = rospy.Time.now()
        clear_marker.action = Marker.DELETEALL
        clear_array = MarkerArray()
        clear_array.markers.append(clear_marker)
        eraser_pub.publish(clear_array)

    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        stamp = rospy.Time.now()
        with lock:
            visible_cloud = maybe_downsample(filtered_cloud, max_points)
            current_occupancy = occupancy
            current_edits = ManualEdits(
                spheres=list(manual_edits.spheres),
                deleted_ids=set(manual_edits.deleted_ids),
            )
            current_deleted_points = (
                points_for_deleted_ids(base_cloud, current_edits.deleted_ids)
                if publish_erasers
                else np.empty((0, 3), dtype=np.float32)
            )
            current_bounds_points = filtered_cloud.xyz.copy()
        publisher.publish(make_pointcloud2(visible_cloud, frame_id, stamp))
        if occupancy_pub is not None and current_occupancy is not None:
            occupancy_pub.publish(make_occupancy_grid(current_occupancy, frame_id, stamp))
        if bounds_pub is not None and len(current_bounds_points) > 0:
            bounds_pub.publish(make_bounds_marker(current_bounds_points, frame_id, stamp))
        if publish_erasers:
            eraser_pub.publish(
                make_eraser_markers(
                    current_edits,
                    current_deleted_points,
                    frame_id,
                    stamp,
                    deleted_point_marker_size,
                )
            )
        rate.sleep()


if __name__ == "__main__":
    main()
