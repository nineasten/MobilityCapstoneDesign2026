# Orin + Phone(Unity APK) + Rokid Max Full-Stack Deployment Spec

This document is a complete, step-by-step instruction set for running the
entire AR navigation pipeline **without a laptop**, using:

- **Jetson AGX Orin** - runs the camera driver, HLoc localization, path
  planner, destination selector, and the `ros_tcp_endpoint` ROS<->Unity
  bridge (everything that today runs on the laptop).
- **Android phone** - runs a Unity-built APK that replaces the laptop's
  Unity Editor session. Connects to the Orin's ROS over WiFi and renders
  the same AR scene (path ribbon, arrows, destination pin, minimap,
  occupancy occlusion) full-screen.
- **Rokid Max glasses** - plugged into the phone via USB-C (DP Alt Mode),
  mirrors the phone's full-screen output. No SDK integration needed - this
  was confirmed by hands-on testing.

**Feasibility verdict: yes, this is achievable with the current codebase.**
The ROS-side pipeline (`live_ar_navigation.launch`) already does everything
needed and `ros_tcp_endpoint` already binds to `0.0.0.0:10000`, so it is
reachable from the phone over WiFi with zero ROS-side code changes. The only
new work is (1) finishing the Orin environment setup, (2) building a new
Unity Android target with retuned camera-rig parameters and a fullscreen
Android scene, and (3) wiring up the WiFi hotspot network. All of this is
described below. Items still unverified end-to-end are flagged in
**Section 6 (Open Risks)**.

---

## 1. Hardware Connection Diagram

```
[Battery] --power--> [Jetson AGX Orin]
                          |
                          |-- USB --> [Arducam IMX219 camera]
                          |             (mounted on/near the Rokid Max,
                          |              attached separately by the user -
                          |              NOT the Rokid Max's own camera)
                          |
                          |-- WiFi (client) --> connects to phone's hotspot
                          |
[Android Phone] --hotspot--> (WiFi AP, Orin connects as client)
       |
       |-- USB-C (DP Alt Mode) --> [Rokid Max glasses]
                                     (mirrors phone screen, optical
                                      see-through AR overlay)
```

Key points:
- The Orin gets power from the battery and reads camera frames over USB from
  the Arducam (separately mounted, independent of the Rokid Max body).
- The phone is physically connected only to the Rokid Max (USB-C, DP Alt
  Mode mirroring - already confirmed working).
- The phone <-> Orin link is **WiFi only**: phone's mobile hotspot is the
  access point, Orin's `wlan0` joins it as a client.
- No extra network hardware (router/AP dongle) is required - the Orin's
  built-in WiFi (`wlan0`) is sufficient.

---

## 2. Orin-Side Setup

### 2.1 Data layout (already transferred)

All large data lives on the external drive:
`/media/acl1/8C12907912906A4A/orin_ar_nav/{data,catkin_ws,venv}`

The ROS launch files hardcode paths under `$(env HOME)/capstonews/...`. Set
up symlinks so those paths resolve onto the external drive (run once):

```bash
mkdir -p ~/capstonews/src ~/capstonews/model_cache ~/capstonews/external
ln -sfn /media/acl1/8C12907912906A4A/orin_ar_nav/data/external/hloc        ~/capstonews/external/hloc
ln -sfn /media/acl1/8C12907912906A4A/orin_ar_nav/data/model_cache/torch    ~/capstonews/model_cache/torch
ln -sfn /media/acl1/8C12907912906A4A/orin_ar_nav/data/src/hloc_reference   ~/capstonews/src/hloc_reference
ln -sfn /media/acl1/8C12907912906A4A/orin_ar_nav/data/src/calibration.json ~/capstonews/src/calibration.json
ln -sfn /media/acl1/8C12907912906A4A/orin_ar_nav/data/src/planning_future_arducam ~/capstonews/src/planning_future_arducam
```

Verify each symlink resolves to a real, non-empty path (especially
`hloc_reference`, which should be ~6.8GB):

```bash
du -sh ~/capstonews/external/hloc ~/capstonews/model_cache/torch \
       ~/capstonews/src/hloc_reference ~/capstonews/src/planning_future_arducam
ls -la ~/capstonews/src/calibration.json
```

### 2.2 Python environment

The venv at `/media/acl1/8C12907912906A4A/orin_ar_nav/venv` should be created
with `--system-site-packages` so it can use the Jetson's apt-installed,
CUDA-enabled OpenCV. **Do not `pip install opencv-python`** - this would
shadow the system build with a CPU-only wheel.

Activate it and check what's missing:

```bash
source /media/acl1/8C12907912906A4A/orin_ar_nav/venv/bin/activate
python -c "import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount())"
python -c "import hloc" 2>&1 | tail -1
python -c "import lightglue" 2>&1 | tail -1
python -c "import pycolmap; print(pycolmap.__version__)" 2>&1 | tail -1
```

If `hloc` / `lightglue` are missing, install them with these **pinned**
commands (matches the laptop's working environment exactly):

```bash
pip install "git+https://github.com/cvg/LightGlue@eb42fee2d71449efb0aa5c10549752b5d75384d8"
pip install --no-deps "git+https://github.com/cvg/Hierarchical-Localization.git@c13273bd0ecc2917a35910fd843712a1c6243193"
pip install gdown kornia==0.7.3 matplotlib plotly scipy tqdm
```

- `--no-deps` on the hloc install is **required** - without it, pip will try
  to pull in `pycolmap>=3.13.0`, which conflicts with the laptop's working
  `pycolmap==3.12.5`. If pycolmap is missing entirely on the Orin, that is a
  separate build/install task - check first with the `import pycolmap` test
  above before doing anything about it.
- These exact commit hashes are what the laptop's environment was built
  from (per `direct_url.json` records) - do not substitute newer commits, as
  the FP16 SuperPoint fix in
  `external/hloc/third_party/SuperGluePretrainedNetwork` depends on this
  LightGlue/hloc version pairing.

### 2.3 catkin build

```bash
cd /media/acl1/8C12907912906A4A/orin_ar_nav/catkin_ws
catkin build   # or catkin_make, matching whatever was used to build the rest of the workspace
source devel/setup.bash
```

Confirm `ros_tcp_endpoint` and all `MobilityCapstoneDesign2026` packages
(`mobility_camera_bringup`, `hloc_localization`, `occupancy_path_planner`,
`destination_selector`) show up in `rospack list` / build without errors.

### 2.4 Camera device path

The launch file's default `video_device` is the **laptop's** USB path. On
the Orin, find the actual device:

```bash
ls -la /dev/v4l/by-id/
```

Look for an Arducam IMX219 entry (likely a different `by-id` string than the
laptop's `usb-Arducam_Arducam_B0495__USB3_2.3MP__Arducam_202500915_0001-video-index0`).
You will pass this as `video_device:=...` when launching (Section 4).

---

## 3. Network Setup (Phone Hotspot -> Orin WiFi)

### 3.1 On the phone
Turn on **mobile hotspot / personal hotspot** (Settings -> Network & internet
-> Hotspot & tethering). Note the SSID and password.

### 3.2 On the Orin
The Orin's WiFi is currently soft-blocked. Enable it and connect (run with a
real terminal - **not** over SSH, since `sudo` needs an interactive TTY/askpass):

```bash
sudo rfkill unblock wifi
nmcli radio wifi on
nmcli device wifi connect "<PHONE_HOTSPOT_SSID>" password "<PHONE_HOTSPOT_PASSWORD>"
```

### 3.3 Determine the Orin's IP on the hotspot network

```bash
ip -4 addr show wlan0
```

Record this IP (e.g. `192.168.x.x`) - it's the address Unity's
`ROSConnection` on the phone needs to target, port `10000`.

### 3.4 Sanity check

From the phone (e.g. via a terminal app, or just proceed to the Unity test in
Section 5) or from another device on the same hotspot:

```bash
ping <orin_wlan0_ip>
```

---

## 4. Running the Pipeline on the Orin

Once 2.x and 3.x are done:

```bash
source /media/acl1/8C12907912906A4A/orin_ar_nav/venv/bin/activate
source /media/acl1/8C12907912906A4A/orin_ar_nav/catkin_ws/devel/setup.bash

roslaunch occupancy_path_planner live_ar_navigation.launch \
  video_device:=/dev/v4l/by-id/<ACTUAL_ORIN_DEVICE_PATH> \
  gui_on_start:=false \
  rviz_on_select:=false
```

Notes:
- `tcp_ip` defaults to `0.0.0.0` and `tcp_port` to `10000` - **no change
  needed**, this is already reachable from the phone over `wlan0`.
- `image_width/height` default to `960x600`, matching the production
  calibration (`arducam_camera_info.yaml`) - leave as default unless the
  Orin-side camera was reconfigured.
- `gui_on_start:=false rviz_on_select:=false` disables RViz, which isn't
  needed when the only consumer is the phone's Unity app. If you want to
  also monitor on the Orin's own display/HDMI output, set these to `true`.
- This single launch brings up: camera driver + config, HLoc localization,
  occupancy path planner, destination selector, voxel map publisher, and
  `ros_tcp_endpoint` - i.e. everything the laptop used to run.

Watch the console output for:
- Camera driver started successfully (no `/dev/video*` errors)
- `hloc_localization` node loads the reference map from
  `~/capstonews/src/hloc_reference` without `ModuleNotFoundError`
- `ros_tcp_endpoint` prints something like `Starting server on 0.0.0.0:10000`

---

## 5. Phone-Side: Unity Android Build

This part is built **on a PC with Unity Editor** (can be the laptop), then
the resulting `.apk` is sideloaded onto the phone. The phone itself does not
need Unity Editor, ROS, or any Python/ML stack - the Unity Editor's Android
Build Support module does the heavy lifting once.

### 5.1 Switch build target

In the `ARPathRenderer` Unity project (`capstonews/unity/ARPathRenderer`):
- `File > Build Settings > Android > Switch Platform`
- Install the **Android Build Support** module (with Android SDK & NDK) via
  Unity Hub if not already installed.

### 5.2 Point ROSConnection at the Orin

In the Unity Editor, set the `ROSConnection` singleton's IP/port to the
Orin's `wlan0` IP from Section 3.3, port `10000`. This setting is not
version-controlled (per `CLAUDE.md`), so it must be set manually for this
build.

### 5.3 Fullscreen / immersive settings

In `Player Settings` for Android:
- Resolution and Presentation -> set to **Landscape** orientation (match
  whatever orientation the Rokid Max mirroring expects - confirm against the
  laptop's monitor orientation used previously).
- Enable **Fullscreen Mode** (Immersive / "Hide navigation bar" if available
  in the target Android API level) so there's no status bar / nav bar
  cutting into the AR view.
- Disable any UI chrome (splash screen can stay default for now).

### 5.4 Retune `RokidSbsCameraRig.cs` for the phone screen

Currently tuned for a 1920x1200 PC monitor tethered via HDMI. For the
phone's screen + Rokid Max mirroring, retune in the Inspector or script
defaults:
- `interpupillaryDistanceMeters` = `0.064` (keep, this is a physical
  human-IPD constant, not display-dependent)
- `convergenceDistanceMeters` = `6.0` (starting point - may need on-device
  tuning for comfort)
- `verticalFieldOfViewDegrees` = `52.6` (starting point - **this is the
  parameter most likely to need adjustment** for the phone's aspect
  ratio/resolution vs the old 1920x1200 target; verify visually once running)

Treat these three as starting values; expect a short visual-tuning pass once
the phone+glasses are actually worn.

### 5.5 Build and install the APK

- `File > Build Settings > Build` -> produces `ARPathRenderer.apk`.
- Transfer to phone (USB cable + `adb install ARPathRenderer.apk`, or any
  file-transfer method) and install (enable "install from unknown sources"
  if needed).

---

## 6. Staged Verification Plan

Do these **in order** - each stage isolates a different failure mode.

1. **Network reachability**
   - Phone hotspot ON, Orin connects (`ip -4 addr show wlan0` shows an IP on
     the hotspot subnet).
   - `ping <orin_ip>` succeeds from another device on the same hotspot.

2. **`ros_tcp_endpoint` reachability (no full pipeline yet)**
   - On Orin: `roslaunch ros_tcp_endpoint endpoint.launch`
   - From the phone's Unity app (or `adb logcat` while it tries to connect):
     confirm a TCP connection to `<orin_ip>:10000` succeeds. This isolates
     "can Unity reach ROS at all" from "does the full pipeline work."

3. **Full ROS pipeline on Orin (no phone yet)**
   - Run `live_ar_navigation.launch` as in Section 4.
   - Confirm topics are publishing: `rostopic hz /planner/current_pose`,
     `rostopic hz /path_planner/unity_path`, `rostopic hz
     /voxel_map/xy_occupancy`, `rostopic hz /arducam_imx219/camera_info`.
   - This re-validates the Orin can sustain the pipeline (the
     `tools/orin/README.md` benchmark task) - if this stage was already
     verified separately, it can be skipped, but it's worth re-confirming
     with the camera now in its final mounted position.

4. **Unity APK display test on phone + glasses (Orin pipeline running)**
   - With stage 3's `roslaunch` running and the phone connected to the
     Orin's hotspot, launch the APK on the phone.
   - Confirm the same visual elements seen on the laptop (path ribbon,
     arrows, destination pin, minimap, occupancy occlusion) render
     correctly full-screen on the phone.
   - Plug phone into Rokid Max via USB-C, confirm mirroring shows the same
     scene through the glasses.

5. **End-to-end integration test**
   - Walk the actual route with camera + Orin + phone + glasses all live.
   - Confirm localization updates (`/planner/current_pose` changing as you
     move), path replanning, and AR overlay all behave as they did on the
     laptop setup.

---

## 7. Open Risks / Outstanding Blockers

These are explicitly **unverified** - flag any failures here back to the
laptop-side session:

- **`pycolmap` on Orin**: status unknown. If `import pycolmap` fails or the
  installed version is `>=3.13.0`, this may conflict with the `--no-deps`
  hloc install. Report the exact version/error before attempting a fix.
- **hloc/lightglue install**: not yet confirmed done on the Orin as of this
  writing. Run the checks in Section 2.2 first.
- **OpenCV CUDA in venv**: confirm `cv2.cuda.getCudaEnabledDeviceCount() > 0`
  inside the venv - if it's 0, the `--system-site-packages` venv may not be
  picking up the Jetson's CUDA-enabled OpenCV build.
- **Orin WiFi**: `rfkill`/`nmcli` steps (Section 3.2) require a local
  terminal session - cannot be done over SSH (no askpass helper configured).
- **`RokidSbsCameraRig` FOV retuning** (Section 5.4): values are carried over
  from the PC-monitor setup and will likely need on-device adjustment.
- **Display orientation**: confirm whether the Rokid Max mirroring expects
  landscape or portrait from the phone - this affects the Unity Player
  Settings in 5.3.
- **Full pipeline performance on battery power**: the Orin benchmark in
  `tools/orin/README.md` may have been run on wall power / a different power
  mode. If running on battery, re-check `nvpmodel` settings per
  `tools/orin/setup_power_mode.sh`.

---

## 8. Quick Reference - Key Files

- Main launch: `occupancy_path_planner/launch/live_ar_navigation.launch`
- HLoc reference data: `~/capstonews/src/hloc_reference/` (symlink to
  external drive)
- Camera calibration (production):
  `mobility_camera_bringup/config/arducam_camera_info.yaml` (960x600)
- ROS<->Unity bridge: `ros_tcp_endpoint` package, `endpoint.launch`
  (`tcp_ip=0.0.0.0`, `tcp_port=10000` by default)
- Unity project (separate repo, no GitHub remote):
  `capstonews/unity/ARPathRenderer`
  - `RosPoseCameraTracker.cs`, `RosPathRibbonRenderer.cs`,
    `RosOccupancyOcclusionRenderer.cs`, `RosTopDownMinimapOverlay.cs`,
    `RosProjectedPathImageOverlay.cs` - all use
    `ROSConnection.GetOrCreateInstance()`
  - `RokidSbsCameraRig.cs` - camera rig params to retune (Section 5.4)
