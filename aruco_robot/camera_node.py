#!/usr/bin/env python3
"""
camera_node.py — runs on the robot (Pi)
Picamera2 capture runs in a dedicated background thread so it never
blocks the ROS2 executor. The ROS timer just grabs the latest frame
and publishes it — even under heavy XGO serial load.
"""

import threading
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from picamera2 import Picamera2

qos_video = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter("width",        1920)
        self.declare_parameter("height",       1080)
        self.declare_parameter("fps",          15)
        self.declare_parameter("jpeg_quality", 40)
        self.declare_parameter("publish_raw",  False)

        self.width        = self.get_parameter("width").value
        self.height       = self.get_parameter("height").value
        self.fps          = self.get_parameter("fps").value
        self.jpeg_quality = self.get_parameter("jpeg_quality").value
        self.pub_raw      = self.get_parameter("publish_raw").value

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
            f"Picamera2 started @ {self.width}x{self.height} {self.fps}fps"
        )

        # ── Publishers ───────────────────────────────────────────────
        self.comp_pub = self.create_publisher(
            CompressedImage, "camera/compressed", qos_video
        )
        if self.pub_raw:
            self.raw_pub = self.create_publisher(
                Image, "camera/image_raw", qos_video
            )

        # ── Background capture thread ────────────────────────────────
        # Stores the latest BGR frame; never blocks the ROS executor.
        self._latest_bgr = None
        self._lock       = threading.Lock()
        self._running    = True

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()

        # ── Publish timer — just grabs latest frame and sends it ─────
        self.timer = self.create_timer(1.0 / self.fps, self._publish_frame)

    # ── Background capture loop ───────────────────────────────────────
    def _capture_loop(self):
        """Runs independently of ROS. Always captures the newest frame."""
        while self._running:
            try:
                frame_rgb = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                with self._lock:
                    self._latest_bgr = frame_bgr
            except Exception as e:
                self.get_logger().error(f"Capture error: {e}")

    # ── ROS publish timer ─────────────────────────────────────────────
    def _publish_frame(self):
        with self._lock:
            frame = self._latest_bgr

        if frame is None:
            return

        now = self.get_clock().now().to_msg()

        # Compressed (JPEG)
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if ok:
            msg = CompressedImage()
            msg.header.stamp    = now
            msg.header.frame_id = "camera"
            msg.format          = "jpeg"
            msg.data            = buf.tobytes()
            self.comp_pub.publish(msg)

        # Raw (optional)
        if self.pub_raw:
            raw = Image()
            raw.header.stamp    = now
            raw.header.frame_id = "camera"
            raw.height          = frame.shape[0]
            raw.width           = frame.shape[1]
            raw.encoding        = "bgr8"
            raw.step            = frame.shape[1] * 3
            raw.data            = frame.tobytes()
            self.raw_pub.publish(raw)

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
