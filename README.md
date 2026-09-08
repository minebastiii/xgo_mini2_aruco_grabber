# ArUco Pick-and-Place Robot (XGO Quadruped)

Autonomous pick-and-place demo on an **XGO quadruped robot** (Raspberry Pi + Picamera2). The robot searches for an ArUco-marked target, walks up to it, picks it up with its claw, walks to an ArUco-marked trailer/container, and drops the object in.

## How it works

The robot runs two ROS2 nodes:

| Node | File | Role |
|---|---|---|
| `camera_node` | `camera_node.py` | Captures frames from the Picamera2, runs ArUco detection + `solvePnP` pose estimation, filters/persists detections, publishes plain camera feed + detection results |
| `fsm_node` | `fsm_node.py` | Finite-state machine that drives the XGO's legs/arm/claw based on the detections published by `camera_node` |

Both nodes run **on the robot** (Raspberry Pi).

### Pipeline

```
Picamera2 → camera_node
              ├─ publishes camera/compressed (+ camera/image_raw, optional)
              ├─ detects ArUco markers (target + trailer, configurable dict/IDs)
              ├─ estimates distance via solvePnP
              ├─ applies EMA filtering + short-term temporal persistence
              └─ publishes aruco/target/* and aruco/trailer/*
                                │
                                ▼
                            fsm_node
              ├─ SEARCH → ALIGN → APPROACH → GRASP → VERIFY
              │                                  │
              │                                  ▼ (target picked up)
              └─ SEARCH → ALIGN → APPROACH → DEPLOY → DONE
                 (now looking for the trailer marker instead of the target)
```

### FSM states

- **SEARCH** — robot turns in place looking for the current marker (target, or trailer once the object is picked up)
- **ALIGN** — turns/straves to center the marker in frame
- **APPROACH** — walks forward, using either measured distance (`solvePnP`) or marker pixel-area as the closing signal
- **GRASP** — claw/arm sequence to pick up the target
- **VERIFY** — backs off and checks the target marker is no longer visible (confirms a successful grasp) before switching mode to search for the trailer
- **DEPLOY** — claw/arm sequence to drop the object into the trailer
- **DONE** — resets pose and returns to **SEARCH** for the next cycle

### Detection details

- Multiple ArUco dictionaries supported simultaneously (4x4 up to 7x7, plus `DICT_ARUCO_ORIGINAL`); each target/trailer entry in the config picks its own dictionary.
- Trailers support a **near marker** (`ids`) and an optional **far-field fallback marker** (`far_ids`) — useful when the primary marker is too small to detect at long range but a larger marker on the same object can be seen first.
- Distance is estimated per-marker via `cv2.solvePnP` using the camera intrinsics and the marker's physical size (`marker_size_m`).
- Detections are smoothed with an EMA filter (`filter_alpha`) and held briefly after the marker briefly drops out of view (`persist_ttl`), so brief occlusions don't reset the FSM.
- Persistence is disabled automatically in `GRASP`/`VERIFY`/`DEPLOY`/`DONE` — the robot must not act on stale detections while it is no longer trying to track a marker.

## Package layout

```
aruco_robot/
├── aruco_robot/
│   ├── camera_node.py
│   └── fsm_node.py
├── launch/
│   └── robot.launch.py
├── config/
│   └── aruco_config.yaml
└── ...
```

Node executables (as referenced by the launch file): `camera_node_new`, `fsm_node_new`.

## Requirements

Runs on a Raspberry Pi on the robot, ROS2, with:
- `picamera2`
- `opencv-contrib-python` (for `cv2.aruco`)
- `numpy`, `pyyaml`
- `xgolib` (XGO SDK — optional at import time; if missing, motor commands are suppressed and a warning is logged instead of crashing)

## Running it

The launch file is startet automatically, via a service, after turning on the robot. If you run multiple robots at the same time don't forget to change the ROS_DOMAIN_ID for each robot.

To stop the service:

```bash
sudo systemctl stop aruco-robot.service
```

To start / restart service:

```bash
sudo systemctl start aruco-robot.service
sudo systemctl restart aruco-robot.service
```

For manual start:

```bash
ros2 launch aruco_robot robot.launch.py
```

See `CHEATSHEET.md` for marker/config setup and launch arguments.

## Topics

**Published by `camera_node`:**
| Topic | Type | Notes |
|---|---|---|
| `camera/compressed` | `sensor_msgs/CompressedImage` | Raw JPEG feed |
| `camera/image_raw` | `sensor_msgs/Image` | Optional, only if `publish_raw:=true` |
| `aruco/target/found` | `std_msgs/Bool` | |
| `aruco/target/data` | `std_msgs/Float32MultiArray` | `[cx, cy, area, marker_id]` |
| `aruco/target/data_filtered` | `std_msgs/Float32MultiArray` | EMA-filtered `[cx, cy, area, marker_id]` |
| `aruco/target/name` | `std_msgs/String` | Matched target name from config |
| `aruco/target/distance` | `std_msgs/Float32` | Meters, from `solvePnP`, `-1.0` if unknown |
| `aruco/trailer/*` | (same shape as `aruco/target/*`) | |
| `camera/params/current` | `std_msgs/String` (JSON) | Current tuning/config, published every 2s |

**Subscribed by `camera_node`:**
| Topic | Type | Notes |
|---|---|---|
| `camera/params` | `std_msgs/String` (JSON) | Live target/trailer config + tuning updates |
| `robot/state` | `std_msgs/String` | FSM state, used to clear temporal persistence on `GRASP`/`VERIFY`/`DEPLOY`/`DONE` |

**Published by `fsm_node`:**
| Topic | Type | Notes |
|---|---|---|
| `robot/state` | `std_msgs/String` | Current FSM state name |
| `fsm/control_mode` | `std_msgs/String` (JSON) | Current mode/tuning snapshot |
| `robot/battery` | `std_msgs/Float32` | Polled every 10s from the XGO SDK |

**Subscribed by `fsm_node`:**
| Topic | Type | Notes |
|---|---|---|
| `aruco/target/data`, `aruco/target/data_filtered`, `aruco/target/distance` | | |
| `aruco/trailer/data`, `aruco/trailer/data_filtered`, `aruco/trailer/distance` | | |
| `fsm/control` | `std_msgs/String` (JSON) | Runtime control (switch distance/area mode, thresholds, approach tuning) |

## Notes

- `camera_node` no longer computes any debug visuals (masks, ROI boxes, overlays) — it only publishes the plain camera feed and detection numbers.
- If the XGO SDK (`xgolib`) is not importable, `fsm_node` still starts but suppresses all motor commands and logs a warning — useful for testing detection/FSM logic off-robot.
