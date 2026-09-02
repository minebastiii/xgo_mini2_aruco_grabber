#!/usr/bin/env python3
"""
camera_node.py — runs on the robot (Pi)

Picamera2 capture + ArUco pose detection laufen in einem dedizierten
Background-Thread, damit weder Capture noch Detection je den ROS2
Executor blockieren — auch nicht unter starker XGO-Serial-Last.

Erkennt ZWEI unabhängige Marker-IDs im selben Bild:
  - target_marker_id  -> das zu greifende Objekt (Würfel)
  - trailer_marker_id -> der Container/Trailer, in den abgelegt wird

Für jede ID wird (falls im Bild sichtbar) publiziert:
  aruco/<target|trailer>/data   (Float32MultiArray: cx, cy, area, id)
  aruco/<target|trailer>/pose   (Float32MultiArray: mx, mz, bearing_deg, id)

"mx"/"mz" sind die Kameraposition im MARKER-eigenen Koordinatensystem:
  mz = Abstand entlang der Markernormale (wie weit noch geradeaus fahren)
  mx = seitlicher Versatz von der Senkrechten durch die Markermitte
mx -> 0 zu regeln ist das, was den Roboter tatsächlich "genau vor" den
Marker stellt — reine Bildzentrierung (cx) regelt nur die Peilung.

WICHTIG: camera_fx/fy/cx/cy sind Platzhalter! Ohne echte Kalibrierung
eurer Picam sind mx/mz nur grob richtig. cv2.calibrateCamera mit einem
Schachbrett/Charuco-Board machen und die Werte hier eintragen.

WICHTIG: target_marker_id und trailer_marker_id MÜSSEN unterschiedlich
sein und auf die tatsächlich verwendeten Marker-IDs eurer Würfel/
Container gesetzt werden (Parameter, Default 0 und 1 sind Platzhalter).
"""

import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from picamera2 import Picamera2

qos_video = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
)


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # ── Parameter ───────────────────────────────────────────────
        self.declare_parameter("width", 1920)
        self.declare_parameter("height", 1080)
        self.declare_parameter("fps", 15)
        self.declare_parameter("jpeg_quality", 40)
        self.declare_parameter("publish_raw", False)

        # ArUco / Pose-Schätzung
        self.declare_parameter("marker_length_m", 0.10)
        self.declare_parameter("aruco_dict", "DICT_4X4_50")
        # Zwei unterschiedliche Marker-IDs — MUSS zu euren echten
        # Würfeln passen, sonst wird nichts (oder das Falsche) erkannt.
        self.declare_parameter("target_marker_id", 0)
        self.declare_parameter("trailer_marker_id", 1)
        # Kamera-Intrinsics — UNBEDINGT auf die echte Kamera kalibrieren!
        self.declare_parameter("camera_fx", 1400.0)
        self.declare_parameter("camera_fy", 1400.0)
        self.declare_parameter("camera_cx", 960.0)
        self.declare_parameter("camera_cy", 540.0)
        self.declare_parameter("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])

        self.width = self.get_parameter("width").value
        self.height = self.get_parameter("height").value
        self.fps = self.get_parameter("fps").value
        self.jpeg_quality = self.get_parameter("jpeg_quality").value
        self.pub_raw = self.get_parameter("publish_raw").value

        self.marker_length = self.get_parameter("marker_length_m").value
        dict_name = self.get_parameter("aruco_dict").value
        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.trailer_marker_id = int(self.get_parameter("trailer_marker_id").value)

        if self.target_marker_id == self.trailer_marker_id:
            self.get_logger().error(
                "target_marker_id == trailer_marker_id — das kann nicht "
                "funktionieren, bitte unterschiedliche IDs setzen!"
            )

        self.camera_matrix = np.array(
            [
                [
                    self.get_parameter("camera_fx").value,
                    0.0,
                    self.get_parameter("camera_cx").value,
                ],
                [
                    0.0,
                    self.get_parameter("camera_fy").value,
                    self.get_parameter("camera_cy").value,
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.array(
            self.get_parameter("dist_coeffs").value, dtype=np.float64
        )

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dict_name)
        )
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.aruco_params)

        # Objektpunkte für solvePnP (Marker-Zentrum = Ursprung, Reihenfolge
        # muss zur cv2.aruco Corner-Reihenfolge TL,TR,BR,BL passen)
        half = self.marker_length / 2.0
        self._obj_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        # ── Picamera2 — same init as original ───────────────────────
        self.picam2 = Picamera2()
        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height)}
                # format intentionally omitted — same as original
            )
        )
        self.picam2.start()
        self.get_logger().info(
            f"Picamera2 started @ {self.width}x{self.height} {self.fps}fps "
            f"(target_id={self.target_marker_id}, trailer_id={self.trailer_marker_id})"
        )

        # ── Publishers ───────────────────────────────────────────────
        self.comp_pub = self.create_publisher(
            CompressedImage, "camera/compressed", qos_video
        )
        if self.pub_raw:
            self.raw_pub = self.create_publisher(Image, "camera/image_raw", qos_video)

        self.target_data_pub = self.create_publisher(
            Float32MultiArray, "aruco/target/data", 10
        )
        self.target_pose_pub = self.create_publisher(
            Float32MultiArray, "aruco/target/pose", 10
        )
        self.trailer_data_pub = self.create_publisher(
            Float32MultiArray, "aruco/trailer/data", 10
        )
        self.trailer_pose_pub = self.create_publisher(
            Float32MultiArray, "aruco/trailer/pose", 10
        )

        # ── Background capture + detection thread ─────────────────────
        self._latest_bgr = None
        self._latest_target_data = None  # (cx, cy, area, id)
        self._latest_target_pose = None  # (mx, mz, bearing_deg, id)
        self._latest_trailer_data = None
        self._latest_trailer_pose = None
        self._lock = threading.Lock()
        self._running = True

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # ── Publish timer — just grabs latest frame/data and sends it ─
        self.timer = self.create_timer(1.0 / self.fps, self._publish_frame)

    # ── Background capture + detect loop ────────────────────────────
    def _capture_loop(self):
        """Runs independently of ROS. Always captures the newest frame
        und (falls sichtbar) Target- sowie Trailer-Pose."""
        while self._running:
            try:
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                target_data, target_pose, trailer_data, trailer_pose = self._detect(
                    frame_bgr
                )

                with self._lock:
                    self._latest_bgr = frame_bgr
                    self._latest_target_data = target_data
                    self._latest_target_pose = target_pose
                    self._latest_trailer_data = trailer_data
                    self._latest_trailer_pose = trailer_pose
            except Exception as e:
                self.get_logger().error(f"Capture/detect error: {e}")

    def _detect(self, frame_bgr):
        """Erkennt alle Marker im Bild und trennt sie nach
        target_marker_id / trailer_marker_id. Bei mehreren Treffern der
        gleichen ID (Würfel-Seiten) wird die mit größter Fläche genommen
        (= näheste/frontalste Seite).

        Returns: (target_data, target_pose, trailer_data, trailer_pose)
        Jedes davon ist entweder ein Tupel oder None, falls nicht sichtbar.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        target_data = target_pose = trailer_data = trailer_pose = None

        if ids is None or len(ids) == 0:
            return target_data, target_pose, trailer_data, trailer_pose

        ids_flat = ids.flatten()

        for wanted_id, is_target in (
            (self.target_marker_id, True),
            (self.trailer_marker_id, False),
        ):
            idxs = np.where(ids_flat == wanted_id)[0]
            if len(idxs) == 0:
                continue

            # größte Fläche unter den Treffern dieser ID auswählen
            areas = [cv2.contourArea(corners[i][0]) for i in idxs]
            best = idxs[int(np.argmax(areas))]
            c = corners[best][0].astype(np.float32)

            cx = float(np.mean(c[:, 0]))
            cy = float(np.mean(c[:, 1]))
            area = float(cv2.contourArea(c))
            data = (cx, cy, area, wanted_id)

            pose = None
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_points,
                c,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.reshape(3)
                # Kameraposition im Marker-Koordinatensystem: -R^T @ t
                cam_in_marker = -R.T @ t
                mx, my, mz = cam_in_marker
                bearing_deg = float(np.degrees(np.arctan2(t[0], t[2])))
                pose = (float(mx), float(mz), bearing_deg, wanted_id)

            if is_target:
                target_data, target_pose = data, pose
            else:
                trailer_data, trailer_pose = data, pose

        return target_data, target_pose, trailer_data, trailer_pose

    # ── ROS publish timer ─────────────────────────────────────────────
    def _publish_frame(self):
        with self._lock:
            frame = self._latest_bgr
            target_data = self._latest_target_data
            target_pose = self._latest_target_pose
            trailer_data = self._latest_trailer_data
            trailer_pose = self._latest_trailer_pose

        if frame is None:
            return

        now = self.get_clock().now().to_msg()

        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
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
            raw.height = frame.shape[0]
            raw.width = frame.shape[1]
            raw.encoding = "bgr8"
            raw.step = frame.shape[1] * 3
            raw.data = frame.tobytes()
            self.raw_pub.publish(raw)

        self._publish_data_pose(
            self.target_data_pub, self.target_pose_pub, target_data, target_pose
        )
        self._publish_data_pose(
            self.trailer_data_pub, self.trailer_pose_pub, trailer_data, trailer_pose
        )

    @staticmethod
    def _publish_data_pose(data_pub, pose_pub, data, pose):
        d_msg = Float32MultiArray()
        d_msg.data = (
            [data[0], data[1], data[2], float(data[3])]
            if data is not None
            else [0.0, 0.0, 0.0, -1.0]
        )
        data_pub.publish(d_msg)

        p_msg = Float32MultiArray()
        p_msg.data = (
            [pose[0], pose[1], pose[2], float(pose[3])]
            if pose is not None
            else [0.0, -1.0, 0.0, -1.0]
        )
        pose_pub.publish(p_msg)

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
