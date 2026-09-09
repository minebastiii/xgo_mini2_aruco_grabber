#!/usr/bin/env python3
"""
camera_node.py — runs on the ROBOT (Pi)

Trimmed down: this node computes ONLY what the FSM needs — marker
detection, pose/distance, temporal persistence, EMA filtering — and
publishes the plain camera feed. It does NOT compute any debug
visuals (masks, ROI boxes, zoomed crops, overlays) any more; that is
now done entirely on the laptop dashboard from the raw image.

Published topics (unchanged names, so fsm_node needs no changes):
  camera/compressed, camera/image_raw (optional)  -> also consumed by
                                                      the laptop dashboard
                                                      to build debug views
  aruco/target/{found,data,data_filtered,name,distance}
  aruco/trailer/{found,data,data_filtered,name,distance}
  camera/params/current   -> laptop dashboard, for display

Subscribed topics:
  camera/params  (String/JSON) — live target/trailer config + FSM-
                                  relevant tuning (filter_alpha,
                                  persist_ttl, camera intrinsics,
                                  marker sizes), sent by the dashboard
  robot/state    (String)      — from fsm_node, clears temporal
                                  persistence on GRASP/VERIFY/DEPLOY/DONE
"""

import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from picamera2 import Picamera2
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

qos_video = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ── ArUco dictionary map ─────────────────────────────────────────────
DICT_MAP = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
DICT_NAMES = sorted(DICT_MAP.keys())


def _make_params():
    """Tuned DetectorParameters for stable detection at distance."""
    if hasattr(cv2.aruco, "DetectorParameters"):
        p = cv2.aruco.DetectorParameters()
    else:
        p = cv2.aruco.DetectorParameters_create()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minCornerDistanceRate = 0.02
    p.minMarkerDistanceRate = 0.02
    p.errorCorrectionRate = 1.0
    p.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX")
        else 1
    )
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    p.cornerRefinementMinAccuracy = 0.1
    return p


def build_detector(dict_name: str):
    dict_id = DICT_MAP.get(dict_name, cv2.aruco.DICT_4X4_50)
    p = _make_params()
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "ArucoDetector"):
            return ("new", cv2.aruco.ArucoDetector(d, p))
        return ("mid", (d, p))
    d = cv2.aruco.Dictionary_get(dict_id)
    return ("old", (d, p))


def run_detector(det_tuple, gray):
    mode, det = det_tuple
    if mode == "new":
        corners, ids, _ = det.detectMarkers(gray)
    else:
        d, p = det
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=p)
    return corners, ids


# ── Config entry helpers ─────────────────────────────────────────────
def entry_ids(e):
    return e.get("ids", []) if isinstance(e, dict) else list(e)


def entry_dict_name(e):
    return e.get("dict", "4x4_50") if isinstance(e, dict) else "4x4_50"


def make_entry(
    ids,
    type_="",
    dict_name="4x4_50",
    marker_size_m=None,
    far_ids=None,
    far_dict=None,
    far_marker_size_m=None,
):
    """Build a normalised marker-group entry dict."""
    entry = {"ids": [int(i) for i in ids], "dict": str(dict_name)}
    if marker_size_m is not None:
        entry["marker_size_m"] = float(marker_size_m)
    if far_ids:
        entry["far_ids"] = [int(i) for i in far_ids]
        entry["far_dict"] = str(far_dict) if far_dict else str(dict_name)
        if far_marker_size_m is not None:
            entry["far_marker_size_m"] = float(far_marker_size_m)
    return entry


# Only the tuning that actually affects what the FSM receives. Mask /
# ROI / visualization tuning now lives entirely on the dashboard.
ROBOT_PARAMS = {
    "filter_alpha": 0.50,
    "persist_ttl": 1.5,
}

# Params forwarded from the dashboard's camera/params topic, besides
# targets/trailers and the ROBOT_PARAMS keys above.
CAMERA_INTRINSIC_KEYS = (
    "cam_fx", "cam_fy", "cam_cx", "cam_cy", "cam_k1", "cam_k2", "cam_p1", "cam_p2",
    "target_marker_size_m", "trailer_marker_size_m",
)


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # ── Camera params ───────────────────────────────────────────
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("fps", 15)
        self.declare_parameter("jpeg_quality", 40)
        self.declare_parameter("publish_raw", False)
        self.width = self.get_parameter("width").value
        self.height = self.get_parameter("height").value
        self.fps = self.get_parameter("fps").value
        self.jpeg_quality = self.get_parameter("jpeg_quality").value
        self.pub_raw = self.get_parameter("publish_raw").value

        # ── FSM-relevant detector tuning ──────────────────────────────
        self.declare_parameter("config_path", "aruco_config.yaml")
        for k, v in ROBOT_PARAMS.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in ROBOT_PARAMS}

        # ── Camera intrinsics / marker sizes ──────────────────────────
        self.declare_parameter("cam_fx", 1400.0)
        self.declare_parameter("cam_fy", 1400.0)
        self.declare_parameter("cam_cx", 960.0)
        self.declare_parameter("cam_cy", 540.0)
        self.declare_parameter("cam_k1", 0.0)
        self.declare_parameter("cam_k2", 0.0)
        self.declare_parameter("cam_p1", 0.0)
        self.declare_parameter("cam_p2", 0.0)
        self.declare_parameter("target_marker_size_m", 0.019)
        self.declare_parameter("trailer_marker_size_m", 0.10)
        self._update_camera_matrix()

        self._det_cache = {}
        config_path = self.get_parameter("config_path").value
        if not os.path.isabs(config_path):
            try:
                pkg_share = get_package_share_directory("aruco_robot")
                config_path = os.path.join(pkg_share, "config", config_path)
            except Exception:
                config_path = os.path.join(
                    os.path.dirname(__file__), "..", "config", config_path
                )
        self._load_config(config_path)

        # EMA + persistence state
        self._filt = {"target": None, "trailer": None}
        self._filt_lost = {"target": False, "trailer": False}
        self._persist = {
            k: {
                "found": False,
                "data": [0.0, 0.0, 0.0, -1.0],
                "name": "",
                "corners": None,
                "dist": -1.0,
                "ts": 0.0,
            }
            for k in ("target", "trailer")
        }
        self._fsm_state = "UNKNOWN"
        self._NO_PERSIST_STATES = {"GRASP", "VERIFY", "DEPLOY", "DONE"}
        self._CLEAR_ON_ENTER = {"GRASP", "VERIFY", "DEPLOY", "DONE"}

        # ── Picamera2 ────────────────────────────────────────────────
        self.picam2 = Picamera2()
        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height)}
            )
        )
        self.picam2.start()
        self.get_logger().info(
            f"Picamera2 started @ {self.width}x{self.height} {self.fps}fps"
        )

        # ── Subscribers ─────────────────────────────────────────────
        self.create_subscription(String, "camera/params", self._cb_params, 10)
        self.create_subscription(String, "robot/state", self._cb_fsm_state, 10)

        # ── Publishers ──────────────────────────────────────────────
        self.comp_pub = self.create_publisher(CompressedImage, "camera/compressed", qos_video)
        if self.pub_raw:
            self.raw_pub = self.create_publisher(Image, "camera/image_raw", qos_video)

        self.pub = {}
        for kind in ("target", "trailer"):
            self.pub[f"{kind}_found"] = self.create_publisher(Bool, f"aruco/{kind}/found", 10)
            self.pub[f"{kind}_data"] = self.create_publisher(Float32MultiArray, f"aruco/{kind}/data", 10)
            self.pub[f"{kind}_data_filt"] = self.create_publisher(Float32MultiArray, f"aruco/{kind}/data_filtered", 10)
            self.pub[f"{kind}_name"] = self.create_publisher(String, f"aruco/{kind}/name", 10)
            self.pub[f"{kind}_distance"] = self.create_publisher(Float32, f"aruco/{kind}/distance", 10)

        self.params_pub = self.create_publisher(String, "camera/params/current", 10)
        self.create_timer(2.0, self._pub_params)

        # ── Background capture thread (never blocks ROS) ───────────────
        self._latest_bgr = None
        self._lock = threading.Lock()
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Single timer: grabs latest frame, detects markers, publishes.
        self.timer = self.create_timer(1.0 / self.fps, self._process_and_publish)

        dicts_in_use = {entry_dict_name(e) for d in (self.targets, self.trailers) for e in d.values()}
        self.get_logger().info(
            f"CameraNode ready. Dicts: {dicts_in_use}  "
            f"Targets: {list(self.targets.keys())}  Trailers: {list(self.trailers.keys())}"
        )

    # ── Capture thread ───────────────────────────────────────────────
    def _capture_loop(self):
        while self._running:
            try:
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self._latest_bgr = frame_bgr
            except Exception as e:
                self.get_logger().error(f"Capture error: {e}")

    # ── Camera matrix ────────────────────────────────────────────────
    def _update_camera_matrix(self):
        fx = self.get_parameter("cam_fx").value
        fy = self.get_parameter("cam_fy").value
        cx = self.get_parameter("cam_cx").value
        cy = self.get_parameter("cam_cy").value
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self._dist = np.array(
            [
                self.get_parameter("cam_k1").value,
                self.get_parameter("cam_k2").value,
                self.get_parameter("cam_p1").value,
                self.get_parameter("cam_p2").value,
                0.0,
            ],
            dtype=np.float64,
        )
        self._target_msize = self.get_parameter("target_marker_size_m").value
        self._trailer_msize = self.get_parameter("trailer_marker_size_m").value

    # ── Pose estimation ───────────────────────────────────────────────
    def _estimate_distance(self, corners_4x2: np.ndarray, marker_size_m: float) -> float:
        half = marker_size_m / 2.0
        obj_pts = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        img_pts = corners_4x2.astype(np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self._K, self._dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                return float(np.linalg.norm(tvec))
        except Exception:
            pass
        return -1.0

    # ── Detector cache ───────────────────────────────────────────────
    def _get_det(self, dict_name: str):
        if dict_name not in self._det_cache:
            self._det_cache[dict_name] = build_detector(dict_name)
        return self._det_cache[dict_name]

    # ── Config ──────────────────────────────────────────────────────
    def _load_config(self, path):
        self.targets = {}
        self.trailers = {}
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            for name, val in cfg.get("targets", {}).items():
                if isinstance(val, dict):
                    self.targets[name] = make_entry(
                        val.get("ids", []), val.get("type", ""), val.get("dict", "4x4_50"),
                        marker_size_m=val.get("marker_size_m"),
                    )
                else:
                    self.targets[name] = make_entry(val)
            for name, val in cfg.get("trailers", {}).items():
                if isinstance(val, dict):
                    self.trailers[name] = make_entry(
                        val.get("ids", []), val.get("type", ""), val.get("dict", "4x4_50"),
                        marker_size_m=val.get("marker_size_m"),
                        far_ids=val.get("far_ids"), far_dict=val.get("far_dict"),
                        far_marker_size_m=val.get("far_marker_size_m"),
                    )
                else:
                    self.trailers[name] = make_entry(val)
            self.get_logger().info(
                f"Config loaded from {path}  targets={list(self.targets.keys())}  "
                f"trailers={list(self.trailers.keys())}"
            )
        except Exception as e:
            self.get_logger().error(f"Config load failed: {e}")

    # ── Live param update from dashboard ─────────────────────────────
    def _cb_params(self, msg: String):
        try:
            updates = json.loads(msg.data)
            for k, v in updates.items():
                if k == "targets":
                    self.targets = {
                        n: make_entry(
                            e.get("ids", []) if isinstance(e, dict) else e,
                            e.get("type", "") if isinstance(e, dict) else "",
                            e.get("dict", "4x4_50") if isinstance(e, dict) else "4x4_50",
                            marker_size_m=e.get("marker_size_m") if isinstance(e, dict) else None,
                        )
                        for n, e in v.items()
                    }
                elif k == "trailers":
                    self.trailers = {
                        n: make_entry(
                            e.get("ids", []) if isinstance(e, dict) else e,
                            e.get("type", "") if isinstance(e, dict) else "",
                            e.get("dict", "4x4_50") if isinstance(e, dict) else "4x4_50",
                            marker_size_m=e.get("marker_size_m") if isinstance(e, dict) else None,
                            far_ids=e.get("far_ids") if isinstance(e, dict) else None,
                            far_dict=e.get("far_dict") if isinstance(e, dict) else None,
                            far_marker_size_m=e.get("far_marker_size_m") if isinstance(e, dict) else None,
                        )
                        for n, e in v.items()
                    }
                elif k in self.p:
                    self.p[k] = type(self.p[k])(v)
                elif k in CAMERA_INTRINSIC_KEYS:
                    self.set_parameters([rclpy.parameter.Parameter(k, value=float(v))])
                    self._update_camera_matrix()
        except Exception as e:
            self.get_logger().error(f"Param update failed: {e}")

    def _pub_params(self):
        msg = String()
        payload = dict(self.p)
        payload["targets"] = self.targets
        payload["trailers"] = self.trailers
        payload["dict_names"] = DICT_NAMES
        # Also report current camera intrinsics + marker sizes so the
        # dashboard can display/tune them (they're only set on this
        # node otherwise, so without this the UI has no value to show).
        payload["cam_fx"] = float(self._K[0, 0])
        payload["cam_fy"] = float(self._K[1, 1])
        payload["cam_cx"] = float(self._K[0, 2])
        payload["cam_cy"] = float(self._K[1, 2])
        payload["cam_k1"] = float(self._dist[0])
        payload["cam_k2"] = float(self._dist[1])
        payload["cam_p1"] = float(self._dist[2])
        payload["cam_p2"] = float(self._dist[3])
        payload["target_marker_size_m"] = float(self._target_msize)
        payload["trailer_marker_size_m"] = float(self._trailer_msize)
        msg.data = json.dumps(payload)
        self.params_pub.publish(msg)

    # ── EMA filter ───────────────────────────────────────────────────
    def _update_filter(self, kind, cx, cy, area, found):
        alpha = self.p["filter_alpha"]
        if not found:
            if self._filt[kind] is not None:
                self._filt_lost[kind] = True
            return self._filt[kind]
        if self._filt[kind] is None or self._filt_lost.get(kind, False):
            self._filt[kind] = [cx, cy, area]
            self._filt_lost[kind] = False
        else:
            f = self._filt[kind]
            f[0] = alpha * cx + (1 - alpha) * f[0]
            f[1] = alpha * cy + (1 - alpha) * f[1]
            f[2] = alpha * area + (1 - alpha) * f[2]
        return self._filt[kind]

    # ── FSM-state tracking & persistence ─────────────────────────────
    def _cb_fsm_state(self, msg: String):
        new_state = msg.data
        if new_state != self._fsm_state and new_state in self._CLEAR_ON_ENTER:
            self._clear_persistence("target")
            self._clear_persistence("trailer")
        self._fsm_state = new_state

    def _clear_persistence(self, kind: str):
        self._persist[kind]["found"] = False
        self._persist[kind]["ts"] = 0.0

    def _apply_persistence(self, kind, found, data, name, corners, dist):
        if self._fsm_state in self._NO_PERSIST_STATES:
            return found, data, name, corners, dist
        now = time.time()
        p = self._persist[kind]
        if found and data[3] >= 0:
            p.update({"found": True, "data": list(data), "name": name, "corners": corners, "dist": dist, "ts": now})
            return found, data, name, corners, dist
        if p["found"] and (now - p["ts"]) < self.p["persist_ttl"]:
            return True, p["data"], p["name"], p["corners"], p["dist"]
        p["found"] = False
        return found, data, name, corners, dist

    # ── Detect on image with all needed dicts ─────────────────────────
    def _detect_all(self, gray) -> dict:
        needed = {entry_dict_name(e) for d in (self.targets, self.trailers) for e in d.values()}
        for e in self.trailers.values():
            fd = e.get("far_dict")
            if fd:
                needed.add(fd)
        result = {}
        for dn in needed:
            corners_list, ids = run_detector(self._get_det(dn), gray)
            dets = []
            if ids is not None:
                for corner, mid in zip(corners_list, ids.flatten()):
                    pts = corner[0]
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    side = float(np.linalg.norm(pts[0] - pts[1]))
                    dets.append((cx, cy, side * side, int(mid), pts))
            result[dn] = dets
        return result

    # ── Match ─────────────────────────────────────────────────────────
    def _match(self, det_by_dict: dict, group_dict: dict, use_fallback: bool = False):
        for name, entry in group_dict.items():
            dn = entry_dict_name(entry)
            for det in det_by_dict.get(dn, []):
                cx, cy, area, mid = det[0], det[1], det[2], det[3]
                corners = det[4] if len(det) > 4 else None
                if int(mid) in [int(i) for i in entry_ids(entry)]:
                    return True, [cx, cy, area, float(mid)], name, corners, entry.get("marker_size_m")
        if use_fallback:
            for name, entry in group_dict.items():
                far_ids = entry.get("far_ids", [])
                if not far_ids:
                    continue
                far_dn = entry.get("far_dict", entry_dict_name(entry))
                for det in det_by_dict.get(far_dn, []):
                    cx, cy, area, mid = det[0], det[1], det[2], det[3]
                    corners = det[4] if len(det) > 4 else None
                    if int(mid) in [int(i) for i in far_ids]:
                        return True, [cx, cy, area, float(mid)], name, corners, entry.get("far_marker_size_m")
        # Not found — no ROI computed on the robot any more, so just
        # report "nothing" (fsm treats found=False as authoritative
        # regardless of the placeholder position values).
        return False, [0.0, 0.0, 0.0, -1.0], "", None, None

    # ── Main per-frame processing (called from timer, not a subscription) ─
    def _process_and_publish(self):
        with self._lock:
            frame = self._latest_bgr
        if frame is None:
            return

        now = self.get_clock().now().to_msg()

        # Publish plain compressed / raw image — the laptop dashboard
        # uses this to compute all debug visuals itself.
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = now
            msg.header.frame_id = "camera"
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self.comp_pub.publish(msg)
        if self.pub_raw:
            raw = Image()
            raw.header.stamp = now
            raw.header.frame_id = "camera"
            raw.height, raw.width = frame.shape[0], frame.shape[1]
            raw.encoding = "bgr8"
            raw.step = frame.shape[1] * 3
            raw.data = frame.tobytes()
            self.raw_pub.publish(raw)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det_by_dict = self._detect_all(gray)

        # Match — targets: primary only; trailers: primary + far fallback.
        t_found, t_data, t_name, t_corners, t_msize = self._match(det_by_dict, self.targets, use_fallback=False)
        tr_found, tr_data, tr_name, tr_corners, tr_msize = self._match(det_by_dict, self.trailers, use_fallback=True)

        t_dist = self._estimate_distance(t_corners, t_msize if t_msize is not None else self._target_msize) \
            if (t_found and t_corners is not None) else -1.0
        tr_dist = self._estimate_distance(tr_corners, tr_msize if tr_msize is not None else self._trailer_msize) \
            if (tr_found and tr_corners is not None) else -1.0

        t_found, t_data, t_name, t_corners, t_dist = self._apply_persistence("target", t_found, t_data, t_name, t_corners, t_dist)
        tr_found, tr_data, tr_name, tr_corners, tr_dist = self._apply_persistence("trailer", tr_found, tr_data, tr_name, tr_corners, tr_dist)

        t_filt = self._update_filter("target", t_data[0], t_data[1], t_data[2], t_found)
        tr_filt = self._update_filter("trailer", tr_data[0], tr_data[1], tr_data[2], tr_found)

        self._publish_result("target", t_found, t_data, t_name, t_filt, t_dist)
        self._publish_result("trailer", tr_found, tr_data, tr_name, tr_filt, tr_dist)

    # ── Publish ───────────────────────────────────────────────────────
    def _publish_result(self, kind, found, data, name, filt, dist=-1.0):
        b = Bool(); b.data = found
        self.pub[f"{kind}_found"].publish(b)
        fa = Float32MultiArray(); fa.data = [float(v) for v in data]
        self.pub[f"{kind}_data"].publish(fa)
        ff = Float32MultiArray()
        ff.data = [float(filt[0]), float(filt[1]), float(filt[2]), float(data[3])] if filt is not None \
            else [0.0, 0.0, 0.0, float(data[3])]
        self.pub[f"{kind}_data_filt"].publish(ff)
        s = String(); s.data = name
        self.pub[f"{kind}_name"].publish(s)
        dm = Float32(); dm.data = float(dist)
        self.pub[f"{kind}_distance"].publish(dm)

    # ── Cleanup ───────────────────────────────────────────────────────
    def destroy_node(self):
        self._running = False
        self._capture_thread.join(timeout=2.0)
        self.picam2.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
