# Cheat Sheet — Configuring & Running the Experiment

Quick reference for setting up markers, config, and launching a run.

## 1. Prepare markers

- **Target marker**: small ArUco marker on the object to pick up.
- **Trailer marker**: marker on the container/trailer the object gets dropped into.
  - Optionally add a **second, larger marker** on the trailer for long-range detection (`far_ids`) — the small marker is used once the robot is close.
- Measure the **physical side length of each marker in meters** — this is required for accurate distance estimation via `solvePnP`.

## 2. Edit `config/aruco_config.yaml`

```yaml
targets:
  cube:
    ids: [0]
    dict: "4x4_50"
    marker_size_m: 0.019
trailers:
  container:
    ids: [8]              # near marker (primary)
    dict: "4x4_50"
    marker_size_m: 0.019
    far_ids: [1]           # far marker (fallback, optional)
    far_dict: "4x4_50"
    far_marker_size_m: 0.10
```

- `ids`: list of marker IDs that count as this target/trailer (usually just one).
- `dict`: ArUco dictionary name. Supported: `4x4_50`/`100`/`250`/`1000`, `5x5_*`, `6x6_*`, `7x7_*`, `original`.
- `marker_size_m`: side length of the marker in meters — **required for correct distance readings**.
- `far_ids` / `far_dict` / `far_marker_size_m`: only needed on trailers, for the optional large fallback marker.
- You can define multiple targets/trailers by name; the FSM just needs at least one of each.

> You can also push config/tuning changes live at runtime over the `camera/params` topic (JSON) instead of editing the YAML and restarting — same keys as above, plus everything in the tuning table below.

## 3. Calibrate the camera (recommended)

Distance estimation depends on accurate intrinsics. Set these launch args (or `camera/params` at runtime) to your camera's actual calibration:

| Param | Meaning |
|---|---|
| `cam_fx`, `cam_fy` | Focal length (pixels) |
| `cam_cx`, `cam_cy` | Principal point (pixels) |
| `cam_k1`, `cam_k2`, `cam_p1`, `cam_p2` | Lens distortion coefficients |

Defaults in the launch file are a rough placeholder (`fx=fy=1400`, `cx=960`, `cy=540`) for a 1920×1080 frame — replace with your own calibration for reliable `grasp_distance_m` / `deploy_distance_m` triggering.

## 4. Key tuning parameters

### Detection (`camera_node`)
| Param | Default | Meaning |
|---|---|---|
| `filter_alpha` | 0.50 | EMA smoothing factor for marker position/area (higher = less smoothing) |
| `persist_ttl` | 1.5s | How long to keep reporting the last-seen detection after the marker disappears |

### FSM (`fsm_node`)
| Param | Default | Meaning |
|---|---|---|
| `use_distance_target` | true | Use `solvePnP` distance (vs. marker pixel-area) to trigger GRASP |
| `use_distance_trailer` | true | Same, for triggering DEPLOY |
| `grasp_distance_m` | 0.15 | Distance threshold to trigger GRASP |
| `deploy_distance_m` | 0.15 | Distance threshold to trigger DEPLOY |
| `target_marker_area` | 22500.0 | Pixel-area trigger threshold for GRASP (only used if `use_distance_target: false`) |
| `container_marker_area` | 60000.0 | Pixel-area trigger threshold for DEPLOY (only used if `use_distance_trailer: false`) |
| `turn_gain` | 30.0 | Turn speed scaling during ALIGN |
| `forward_gain` | 3.75 | Forward-motion duration scaling (area mode) |
| `forward_speed` | 10.0 | Forward walking speed during APPROACH |
| `cx_threshold` | 0.15 | Allowed horizontal centering error (fraction of frame width) before considered aligned |
| `min_turn` | 7.0 | Minimum turn command magnitude |
| `approach_dist_gain` | 10.0 | s/m — converts remaining distance into a forward-step duration (distance mode) |
| `approach_step_min` / `approach_step_max` | 1.0 / 3.0 | Clamp on forward-step duration (distance mode) |

You can change `use_distance_target`, `use_distance_trailer`, `grasp_distance_m`, `deploy_distance_m`, `approach_dist_gain`, `approach_step_min`, `approach_step_max` live at runtime by publishing JSON to `fsm/control`.

## 5. Launch

```bash
ros2 launch aruco_robot robot_new.launch.py
```

Override any launch argument as needed, e.g.:

```bash
ros2 launch aruco_robot robot_new.launch.py \
    width:=1920 height:=1080 fps:=15 jpeg_quality:=60 \
    config_path:=aruco_config.yaml \
    cam_fx:=1400.0 cam_fy:=1400.0 cam_cx:=960.0 cam_cy:=540.0 \
    target_marker_size_m:=0.019 trailer_marker_size_m:=0.10
```

| Launch arg | Default | Meaning |
|---|---|---|
| `width` / `height` | 1920 / 1080 | Camera resolution |
| `fps` | 15 | Capture/publish rate |
| `jpeg_quality` | 60 | JPEG compression quality for `camera/compressed` |
| `config_path` | `aruco_config.yaml` | Path to marker config (relative to package share `config/` if not absolute) |
| `cam_fx`, `cam_fy`, `cam_cx`, `cam_cy` | see above | Camera intrinsics |
| `target_marker_size_m` | 0.019 | Fallback target marker size if not set per-entry in config |
| `trailer_marker_size_m` | 0.10 | Fallback trailer marker size if not set per-entry in config |
