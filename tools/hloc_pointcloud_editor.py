#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np


@dataclass
class ManualEdits:
    spheres: list
    deleted_ids: Set[int]


@dataclass
class PreparedCloud:
    points: np.ndarray
    colors: Optional[np.ndarray]
    ids: np.ndarray


def parse_args() -> argparse.Namespace:
    workspace = Path.home() / "capstonews"
    reference_dir = workspace / "src" / "hloc_reference"

    parser = argparse.ArgumentParser(
        description="Open3D editor for removing individual HLoc reference point-cloud points."
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=reference_dir / "debug" / "sparse_points_raw.ply",
        help="Input HLoc/COLMAP sparse PLY.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=workspace / "src" / "calibration.json",
        help="Map calibration JSON used by the ROS publisher.",
    )
    parser.add_argument(
        "--eraser-file",
        type=Path,
        default=reference_dir / "debug" / "manual_erasers.json",
        help="Manual edit JSON shared with the ROS publisher.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Display raw COLMAP coordinates instead of applying calibration.json.",
    )
    parser.add_argument(
        "--no-floor-align",
        action="store_true",
        help="Do not auto-align the fitted floor plane to z=0.",
    )
    parser.add_argument(
        "--cloud-min-z",
        type=float,
        default=0.0,
        help="Hide points below this z after calibration/floor alignment. Use --cloud-min-z -inf to disable.",
    )
    parser.add_argument(
        "--cloud-max-z",
        type=float,
        default=None,
        help="Hide points above this z after calibration/floor alignment.",
    )
    parser.add_argument(
        "--floor-fit-min-percentile",
        type=float,
        default=1.0,
        help="Lower z percentile used for floor-plane fitting.",
    )
    parser.add_argument(
        "--floor-fit-max-percentile",
        type=float,
        default=25.0,
        help="Upper z percentile used for floor-plane fitting.",
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=0,
        help="Show at most this many points. 0 shows all editable points.",
    )
    parser.add_argument(
        "--export-cleaned-ply",
        type=Path,
        default=None,
        help="Optionally export the currently transformed/filtered cleaned point cloud.",
    )
    parser.add_argument(
        "--clear-edits",
        action="store_true",
        help="Clear saved manual point deletions and eraser spheres, then exit.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Reopen the editor after each save so you can inspect the updated result.",
    )
    parser.add_argument(
        "--legacy-picker",
        action="store_true",
        help="Use Open3D's built-in Shift+click picker instead of the instant red-click editor.",
    )
    parser.add_argument(
        "--pick-pixel-radius",
        type=float,
        default=12.0,
        help="Maximum screen-pixel distance from the cursor to a point when using the instant editor.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Displayed point size in the Open3D editor.",
    )
    parser.add_argument(
        "--no-guides",
        action="store_true",
        help="Hide the floor grid, origin axes, and cloud bounding box in the instant editor.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=1.0,
        help="Spacing, in map units, for the z=0 floor guide grid.",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=1.5,
        help="Displayed size of the origin XYZ axis guide.",
    )
    return parser.parse_args()


def import_open3d():
    try:
        import open3d as o3d
    except ImportError:
        print(
            "Open3D is not installed.\n"
            "Install it with:\n\n"
            "  python3 -m pip install --user open3d\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return o3d


def load_manual_edits(path: Path) -> ManualEdits:
    if not path.exists():
        return ManualEdits(spheres=[], deleted_ids=set())

    data = json.loads(path.read_text(encoding="utf-8"))
    spheres = data.get("spheres", []) if isinstance(data, dict) else []
    deleted_ids = set()
    if isinstance(data, dict):
        deleted_ids = {int(value) for value in data.get("deleted_ids", [])}
    return ManualEdits(spheres=spheres, deleted_ids=deleted_ids)


def save_manual_edits(path: Path, edits: ManualEdits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spheres": edits.spheres,
        "deleted_ids": sorted(int(value) for value in edits.deleted_ids),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_calibration(points: np.ndarray, calibration_path: Path) -> np.ndarray:
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    scale = float(data.get("scale", 1.0))
    rotation = np.asarray(
        data.get("rotation_matrix", np.eye(3)),
        dtype=np.float64,
    ).reshape(3, 3)
    translation = np.asarray(
        data.get("translation", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)
    return scale * (points @ rotation.T) + translation


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
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * (
        (1.0 - dot) / np.dot(cross, cross)
    )


def align_floor(
    points: np.ndarray,
    min_percentile: float,
    max_percentile: float,
) -> np.ndarray:
    low = float(np.percentile(points[:, 2], min_percentile))
    high = float(np.percentile(points[:, 2], max_percentile))
    floor_points = points[(points[:, 2] >= low) & (points[:, 2] <= high)]
    if len(floor_points) < 1000:
        print("Floor alignment skipped because too few floor candidates were found.")
        return points

    centroid = floor_points.mean(axis=0)
    _, _, vh = np.linalg.svd(floor_points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal

    rotation = rotation_from_vectors(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    aligned = (points - centroid) @ rotation.T + centroid
    aligned_floor = (floor_points - centroid) @ rotation.T + centroid
    floor_z = float(np.median(aligned_floor[:, 2]))
    aligned[:, 2] -= floor_z

    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))
    print(
        "Floor alignment: "
        f"candidates={len(floor_points)} z_window=[{low:.3f}, {high:.3f}] "
        f"tilt={tilt:.2f}deg z_shift={floor_z:.3f}"
    )
    return aligned


def apply_sphere_edits(cloud: PreparedCloud, spheres: list) -> PreparedCloud:
    if not spheres:
        return cloud

    mask = np.ones(len(cloud.points), dtype=bool)
    for sphere in spheres:
        if "center" in sphere:
            center = np.asarray(sphere["center"], dtype=np.float64).reshape(3)
        else:
            center = np.asarray([sphere["x"], sphere["y"], sphere["z"]], dtype=np.float64)
        radius = float(sphere["radius"])
        diff = cloud.points - center.reshape(1, 3)
        mask &= np.einsum("ij,ij->i", diff, diff) > radius * radius

    colors = cloud.colors[mask] if cloud.colors is not None else None
    return PreparedCloud(points=cloud.points[mask], colors=colors, ids=cloud.ids[mask])


def prepare_cloud(o3d, args: argparse.Namespace, edits: ManualEdits) -> PreparedCloud:
    if not args.ply.exists():
        raise FileNotFoundError(args.ply)

    point_cloud = o3d.io.read_point_cloud(str(args.ply))
    points = np.asarray(point_cloud.points, dtype=np.float64)
    colors = None
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors, dtype=np.float64)
    ids = np.arange(len(points), dtype=np.int64)

    if not args.no_calibration:
        if not args.calibration.exists():
            raise FileNotFoundError(args.calibration)
        points = apply_calibration(points, args.calibration)

    if not args.no_floor_align:
        points = align_floor(
            points,
            min_percentile=args.floor_fit_min_percentile,
            max_percentile=args.floor_fit_max_percentile,
        )

    mask = np.ones(len(points), dtype=bool)
    if args.cloud_min_z is not None and not math.isinf(args.cloud_min_z):
        mask &= points[:, 2] >= args.cloud_min_z
    if args.cloud_max_z is not None:
        mask &= points[:, 2] <= args.cloud_max_z
    if edits.deleted_ids:
        mask &= ~np.isin(ids, list(edits.deleted_ids))

    filtered_colors = colors[mask] if colors is not None else None
    cloud = PreparedCloud(points=points[mask], colors=filtered_colors, ids=ids[mask])
    cloud = apply_sphere_edits(cloud, edits.spheres)

    if args.max_display_points > 0 and len(cloud.points) > args.max_display_points:
        indices = np.linspace(
            0,
            len(cloud.points) - 1,
            args.max_display_points,
            dtype=np.int64,
        )
        sampled_colors = cloud.colors[indices] if cloud.colors is not None else None
        cloud = PreparedCloud(
            points=cloud.points[indices],
            colors=sampled_colors,
            ids=cloud.ids[indices],
        )
        print(
            "Warning: max-display-points is active; only displayed sampled points can be deleted."
        )

    print(f"Editable points shown: {len(cloud.points)}")
    return cloud


def to_open3d_cloud(o3d, cloud: PreparedCloud):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(cloud.points)
    if cloud.colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(cloud.colors)
    return point_cloud


def cloud_xy_extent(points: np.ndarray, padding: float) -> Tuple[float, float, float, float]:
    min_x = float(np.min(points[:, 0]) - padding)
    max_x = float(np.max(points[:, 0]) + padding)
    min_y = float(np.min(points[:, 1]) - padding)
    max_y = float(np.max(points[:, 1]) + padding)
    return min_x, max_x, min_y, max_y


def make_floor_grid(o3d, points: np.ndarray, step: float):
    step = max(0.05, float(step))
    min_x, max_x, min_y, max_y = cloud_xy_extent(points, padding=step)
    min_x = math.floor(min_x / step) * step
    max_x = math.ceil(max_x / step) * step
    min_y = math.floor(min_y / step) * step
    max_y = math.ceil(max_y / step) * step

    vertices = []
    lines = []
    colors = []

    def add_line(start, end, color):
        line_index = len(vertices)
        vertices.extend([start, end])
        lines.append([line_index, line_index + 1])
        colors.append(color)

    major_color = [0.70, 0.74, 0.78]
    minor_color = [0.32, 0.35, 0.38]
    axis_x_color = [0.90, 0.18, 0.18]
    axis_y_color = [0.18, 0.72, 0.28]

    x_values = np.arange(min_x, max_x + step * 0.5, step)
    y_values = np.arange(min_y, max_y + step * 0.5, step)
    for x in x_values:
        rounded = int(round(x / step))
        color = major_color if rounded % 5 == 0 else minor_color
        if abs(x) < step * 0.25:
            color = axis_y_color
        add_line([float(x), min_y, 0.0], [float(x), max_y, 0.0], color)

    for y in y_values:
        rounded = int(round(y / step))
        color = major_color if rounded % 5 == 0 else minor_color
        if abs(y) < step * 0.25:
            color = axis_x_color
        add_line([min_x, float(y), 0.0], [max_x, float(y), 0.0], color)

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return grid


def make_cloud_box(o3d, points: np.ndarray):
    bounds = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.min(points, axis=0),
        max_bound=np.max(points, axis=0),
    )
    bounds.color = [0.95, 0.82, 0.20]
    return bounds


def add_visual_guides(o3d, visualizer, points: np.ndarray, grid_step: float, axis_size: float) -> None:
    visualizer.add_geometry(make_floor_grid(o3d, points, grid_step), reset_bounding_box=False)
    visualizer.add_geometry(make_cloud_box(o3d, points), reset_bounding_box=False)
    visualizer.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(0.1, float(axis_size)),
            origin=[0.0, 0.0, 0.0],
        ),
        reset_bounding_box=False,
    )


def set_editor_view(visualizer, points: np.ndarray, preset: str = "angled") -> bool:
    view = visualizer.get_view_control()
    center = np.mean(points, axis=0)
    extent = np.max(np.ptp(points, axis=0))
    zoom = 0.60 if extent > 12.0 else 0.72
    view.set_lookat(center.tolist())

    if preset == "top":
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, 1.0, 0.0])
        view.set_zoom(zoom)
    elif preset == "side":
        view.set_front([1.0, 0.0, -0.08])
        view.set_up([0.0, 0.0, 1.0])
        view.set_zoom(zoom)
    else:
        view.set_front([0.45, -0.55, -0.70])
        view.set_up([0.0, 0.0, 1.0])
        view.set_zoom(zoom)

    visualizer.update_renderer()
    return True


def find_nearest_screen_point(
    visualizer,
    points: np.ndarray,
    mouse_x: float,
    mouse_y: float,
    max_pixel_distance: float,
) -> Optional[int]:
    parameters = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
    intrinsic = parameters.intrinsic.intrinsic_matrix
    extrinsic = parameters.extrinsic

    homogeneous = np.column_stack(
        [points, np.ones(len(points), dtype=np.float64)]
    )
    camera_points = (extrinsic @ homogeneous.T).T[:, :3]
    valid = camera_points[:, 2] > 1e-6
    if not np.any(valid):
        return None

    projected = (intrinsic @ camera_points[valid].T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    deltas = pixels - np.asarray([[mouse_x, mouse_y]], dtype=np.float64)
    distance_squared = np.einsum("ij,ij->i", deltas, deltas)
    nearest_local = int(np.argmin(distance_squared))
    nearest_distance = float(math.sqrt(distance_squared[nearest_local]))
    if nearest_distance > max_pixel_distance:
        print(
            f"No point selected: nearest screen point is {nearest_distance:.1f}px away "
            f"(limit {max_pixel_distance:.1f}px)."
        )
        return None

    valid_indices = np.flatnonzero(valid)
    return int(valid_indices[nearest_local])


def instant_red_click_editor(
    o3d,
    cloud: PreparedCloud,
    point_size: float,
    pick_pixel_radius: float,
    show_guides: bool,
    grid_step: float,
    axis_size: float,
) -> List[int]:
    print("")
    print("Open3D instant point editor")
    print("  Left click: mark nearest point red")
    print("  U: undo last red mark")
    print("  C: clear red marks from this session")
    print("  1/T: top view, 2/S: side view, 3/R: angled overview")
    print("  Arrow keys or H/J/K/L: rotate view")
    print("  Q or Esc: close window and save red-marked points as deletions")
    print("")

    point_cloud = to_open3d_cloud(o3d, cloud)
    if cloud.colors is None:
        original_colors = np.ones((len(cloud.points), 3), dtype=np.float64)
    else:
        original_colors = cloud.colors.copy()
    working_colors = original_colors.copy()
    point_cloud.colors = o3d.utility.Vector3dVector(working_colors)

    selected_order: List[int] = []
    selected_set: Set[int] = set()
    mouse_position = {"x": 0.0, "y": 0.0}

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="HLoc Point Cloud Editor", width=1280, height=800)
    visualizer.add_geometry(point_cloud)
    if show_guides:
        add_visual_guides(
            o3d,
            visualizer,
            cloud.points,
            grid_step=grid_step,
            axis_size=axis_size,
        )
    render_option = visualizer.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([0.18, 0.19, 0.20])

    def refresh_colors(vis) -> bool:
        point_cloud.colors = o3d.utility.Vector3dVector(working_colors)
        vis.update_geometry(point_cloud)
        return True

    def on_mouse_move(_vis, x: float, y: float) -> bool:
        mouse_position["x"] = x
        mouse_position["y"] = y
        return False

    def on_mouse_button(vis, button: int, action: int, _mods: int) -> bool:
        left_button = 0
        press = 1
        if button != left_button or action != press:
            return False

        index = find_nearest_screen_point(
            vis,
            cloud.points,
            mouse_position["x"],
            mouse_position["y"],
            pick_pixel_radius,
        )
        if index is None:
            return False
        if index in selected_set:
            print(f"Point id={int(cloud.ids[index])} is already marked red.")
            return False

        selected_set.add(index)
        selected_order.append(index)
        working_colors[index] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        print(
            f"Marked point id={int(cloud.ids[index])} red "
            f"({len(selected_order)} selected this session)."
        )
        return refresh_colors(vis)

    def undo_last(vis) -> bool:
        if not selected_order:
            print("Nothing to undo.")
            return False
        index = selected_order.pop()
        selected_set.discard(index)
        working_colors[index] = original_colors[index]
        print(f"Undo point id={int(cloud.ids[index])}.")
        return refresh_colors(vis)

    def clear_session(vis) -> bool:
        if not selected_order:
            print("No session marks to clear.")
            return False
        for index in selected_order:
            working_colors[index] = original_colors[index]
        selected_order.clear()
        selected_set.clear()
        print("Cleared all red marks from this session.")
        return refresh_colors(vis)

    def switch_view(preset: str, label: str):
        def callback(vis, action: int = 1, _mods: int = 0) -> bool:
            glfw_press = 1
            glfw_repeat = 2
            if action not in {glfw_press, glfw_repeat}:
                return False
            print(f"View: {label}")
            return set_editor_view(vis, cloud.points, preset)

        return callback

    def rotate_view(delta_x: float, delta_y: float):
        def callback(vis, action: int = 1, _mods: int = 0) -> bool:
            glfw_press = 1
            glfw_repeat = 2
            if action not in {glfw_press, glfw_repeat}:
                return False
            vis.get_view_control().rotate(delta_x, delta_y)
            vis.update_renderer()
            return False

        return callback

    def set_initial_view_once():
        pending = {"value": True}

        def callback(vis) -> bool:
            if pending["value"]:
                set_editor_view(vis, cloud.points, "angled")
                pending["value"] = False
            return False

        return callback

    visualizer.register_mouse_move_callback(on_mouse_move)
    visualizer.register_mouse_button_callback(on_mouse_button)
    visualizer.register_key_callback(ord("U"), undo_last)
    visualizer.register_key_callback(ord("C"), clear_session)
    visualizer.register_key_action_callback(ord("1"), switch_view("top", "top"))
    visualizer.register_key_action_callback(ord("2"), switch_view("side", "side"))
    visualizer.register_key_action_callback(ord("3"), switch_view("angled", "angled"))
    visualizer.register_key_action_callback(ord("T"), switch_view("top", "top"))
    visualizer.register_key_action_callback(ord("S"), switch_view("side", "side"))
    visualizer.register_key_action_callback(ord("R"), switch_view("angled", "angled"))
    visualizer.register_key_action_callback(263, rotate_view(-60.0, 0.0))
    visualizer.register_key_action_callback(262, rotate_view(60.0, 0.0))
    visualizer.register_key_action_callback(265, rotate_view(0.0, -60.0))
    visualizer.register_key_action_callback(264, rotate_view(0.0, 60.0))
    visualizer.register_key_action_callback(ord("H"), rotate_view(-60.0, 0.0))
    visualizer.register_key_action_callback(ord("L"), rotate_view(60.0, 0.0))
    visualizer.register_key_action_callback(ord("K"), rotate_view(0.0, -60.0))
    visualizer.register_key_action_callback(ord("J"), rotate_view(0.0, 60.0))
    visualizer.register_animation_callback(set_initial_view_once())
    visualizer.run()
    visualizer.destroy_window()
    return selected_order


def pick_points(o3d, cloud: PreparedCloud) -> List[int]:
    print("")
    print("Open3D point editor")
    print("  Shift + left click: pick point")
    print("  Shift + right click: undo pick")
    print("  Q or Esc: close window and save picked deletions")
    print("")

    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name="HLoc Point Cloud Editor", width=1280, height=800)
    visualizer.add_geometry(to_open3d_cloud(o3d, cloud))
    render_option = visualizer.get_render_option()
    render_option.point_size = 3.0
    render_option.background_color = np.asarray([0.18, 0.19, 0.20])
    visualizer.run()
    picked = list(visualizer.get_picked_points())
    visualizer.destroy_window()
    return picked


def export_cleaned_cloud(o3d, args: argparse.Namespace, edits: ManualEdits) -> None:
    if args.export_cleaned_ply is None:
        return
    export_args = argparse.Namespace(**vars(args))
    export_args.max_display_points = 0
    cloud = prepare_cloud(o3d, export_args, edits)
    output_cloud = to_open3d_cloud(o3d, cloud)
    args.export_cleaned_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.export_cleaned_ply), output_cloud, write_ascii=False)
    print(f"Exported cleaned transformed PLY: {args.export_cleaned_ply}")


def main() -> int:
    args = parse_args()
    args.ply = args.ply.expanduser().resolve()
    args.calibration = args.calibration.expanduser().resolve()
    args.eraser_file = args.eraser_file.expanduser().resolve()
    if args.export_cleaned_ply is not None:
        args.export_cleaned_ply = args.export_cleaned_ply.expanduser().resolve()

    if args.clear_edits:
        save_manual_edits(args.eraser_file, ManualEdits(spheres=[], deleted_ids=set()))
        print(f"Cleared manual edits: {args.eraser_file}")
        return 0

    o3d = import_open3d()
    edits = load_manual_edits(args.eraser_file)
    print(
        "Loaded manual edits: "
        f"spheres={len(edits.spheres)} deleted_points={len(edits.deleted_ids)}"
    )

    while True:
        cloud = prepare_cloud(o3d, args, edits)
        if len(cloud.points) == 0:
            print("No points to edit after filters/manual edits.")
            break

        if args.legacy_picker:
            picked_indices = pick_points(o3d, cloud)
        else:
            picked_indices = instant_red_click_editor(
                o3d,
                cloud,
                point_size=args.point_size,
                pick_pixel_radius=args.pick_pixel_radius,
                show_guides=not args.no_guides,
                grid_step=args.grid_step,
                axis_size=args.axis_size,
            )
        picked_ids = {int(cloud.ids[index]) for index in picked_indices}
        new_ids = picked_ids - edits.deleted_ids
        edits.deleted_ids.update(new_ids)
        save_manual_edits(args.eraser_file, edits)

        print(
            f"Picked {len(picked_indices)} points, added {len(new_ids)} new deletions."
        )
        print(f"Saved manual edits: {args.eraser_file}")

        if not args.repeat:
            break

        answer = input("Reopen with updated cloud? [Y/n] ").strip().lower()
        if answer in {"n", "no"}:
            break

    export_cleaned_cloud(o3d, args, edits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
