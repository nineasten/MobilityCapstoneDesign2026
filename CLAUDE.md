# AR Navigation Pipeline - Project Context for Claude

This repo is the ROS1 (Noetic) source for an AR navigation pipeline:
camera (Arducam IMX219) -> HLoc visual localization -> occupancy path
planner -> destination selector -> Unity AR renderer on Rokid Max glasses
(optical see-through, PC-tethered via HDMI).

## Current status (as of this commit)

- Pipeline fully optimized and validated on a laptop with RTX3060 Laptop
  GPU (80W cap): full ROS bag playback through `hloc_localization` runs at
  ~0.276s/frame avg (~3.6Hz), 94-96/121 frames accepted, avg ~90-94 inliers.
  Target is 0.5Hz, so there is 2x-6x margin.
- FP16 inference applied to NetVLAD + SuperPoint backbones (`.half()`), and
  to the LightGlue matcher via `torch.autocast` (see
  `hloc_localization/scripts/hloc_persistent_localization_node.py`,
  `match_pairs()`).
- Camera/Unity calibration is for 960x600 (fx=610.129, fy=614.460,
  cx=476.029, cy=297.767); see `src/calibration.json` in the parent
  `capstonews` workspace and `mobility_camera_bringup/config/arducam_camera_info.yaml`.
- Unity AR renderer (`capstonews/unity/ARPathRenderer`) shows path objects
  only (ribbon, arrows, destination pin, minimap, occupancy occlusion) -
  Rokid Max is optical see-through, no camera backdrop.

## Orin migration (in progress)

Target board: Jetson AGX Orin 64GB Developer Kit. Goal: confirm the laptop
performance/power numbers hold (or improve) on Orin, then run the full
pipeline there instead of the laptop.

**Start here**: [`tools/orin/README.md`](tools/orin/README.md) - power mode
setup, ONNX export (already done, see `model_cache/onnx/` in the parent repo),
TensorRT/DLA/nvjpeg checklist.

### What still needs to happen on the Orin
1. Run `tools/orin/setup_power_mode.sh` (verify mode IDs with `nvpmodel -q
   --verbose` first - they vary by JetPack version). Start at 30W, not MAXN.
2. Re-run the same full-pipeline bag benchmark as on the laptop (roslaunch
   `hloc_localization.launch` on `/bench/*` topics + `rosbag play`) to get
   real Orin numbers - don't assume the laptop estimates are exact.
3. Only if the Orin numbers are tight against 0.5Hz: follow the TensorRT/DLA
   steps in `tools/orin/README.md` (build engines ON the Orin - they are not
   portable from the laptop's RTX3060).

## Data NOT in this git repo (transfer separately, e.g. rsync/tarball)

These live in the parent `capstonews` workspace and are gitignored due to
size - they must be copied to the Orin manually:
- `external/hloc/` (126MB, vendored HLoc + SuperGlue/LightGlue third_party,
  **includes the FP16 `grid_sample` dtype fix** in
  `external/hloc/third_party/SuperGluePretrainedNetwork/models/superpoint.py`
  - this fix IS tracked in the `capstonews` repo despite the dir being
  gitignored, via `git add -f`)
- `model_cache/` (torch hub weights + `model_cache/onnx/*.onnx` exported for
  TensorRT prep)
- `src/hloc_reference/` (6.8GB, SfM reference map - required for localization)
- `src/calibration.json`, `src/planning_future_arducam/` (small, tracked in
  `capstonews` repo)

## Things explicitly deferred

- Rokid Max IMU integration (GY-9250/MPU9250 considered) - blocked on
  current HDMI-only tether not carrying IMU data; deferred until after Orin
  validation.

## Conventions / preferences

- Explain things in Korean when discussing with the user (한글로 설명).
- Be honest about what has/hasn't been directly tested (e.g. distinguish
  "isolated benchmark" vs "full ROS pipeline with bag playback").
- Don't make risky/destructive changes without confirming first.
