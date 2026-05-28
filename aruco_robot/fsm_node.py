#!/usr/bin/env python3
"""
fsm_node.py — based strictly on the original ArucoFSM structure.
Strafe is added as substep 0+1 inside DEPLOY (same pattern as GRASP substeps).
No extra states added.
"""

import rclpy
import time
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool, Float32MultiArray, String
from enum import Enum, auto

try:
    from xgolib import XGO
    XGO_AVAILABLE = True
except ImportError:
    XGO_AVAILABLE = False


class State(Enum):
    SEARCH           = auto()
    SEARCH_TURNING   = auto()
    SEARCH_WAIT      = auto()
    ALIGN            = auto()
    ALIGN_TURNING    = auto()
    ALIGN_WAIT       = auto()
    APPROACH         = auto()
    APPROACH_FORWARD = auto()
    APPROACH_BACK    = auto()
    GRASP            = auto()
    VERIFY           = auto()
    DEPLOY           = auto()
    DONE             = auto()


class ArucoFSM(Node):
    def __init__(self):
        super().__init__('aruco_fsm')

        # Parameters
        self.declare_parameter('target_marker_area',    22500.0)
        self.declare_parameter('container_marker_area', 60000.0)
        self.declare_parameter('turn_gain',             30.0)
        self.declare_parameter('forward_gain',          5.0)
        self.declare_parameter('forward_speed',         10.0)
        self.declare_parameter('cx_threshold',          0.15)
        self.declare_parameter('min_turn',              7.0)
        self.declare_parameter('image_width',           1920)
        self.declare_parameter('strafe_left_speed',     10.0)
        self.declare_parameter('strafe_left_duration',  0.6)

        self.target_area           = self.get_parameter('target_marker_area').value
        self.container_target_area = self.get_parameter('container_marker_area').value
        self.turn_gain             = self.get_parameter('turn_gain').value
        self.forward_gain          = self.get_parameter('forward_gain').value
        self.forward_speed         = self.get_parameter('forward_speed').value
        self.cx_threshold          = self.get_parameter('cx_threshold').value
        self.min_turn              = self.get_parameter('min_turn').value
        self.image_width           = float(self.get_parameter('image_width').value)
        self.strafe_left_speed     = self.get_parameter('strafe_left_speed').value
        self.strafe_left_duration  = self.get_parameter('strafe_left_duration').value
        self.min_strave            = 5.0
        self.strave_gain           = 10.0

        # XGO
        if XGO_AVAILABLE:
            self.xgo = XGO(port='/dev/ttyAMA0')
            self.xgo.stop()
            self.xgo.translation("z", 80)
            self.xgo.attitude("p", 15)
        else:
            self.xgo = None
            self.get_logger().warn("XGO not available — motor commands suppressed")

        # FSM
        self.state               = State.SEARCH
        self.prev_state          = None
        self.motion_active       = False
        self.substep             = 0
        self.searching_container = False
        self.picked_up_id        = -1
        self.get_logger().info("[FSM] Starting")

        # Detection data — raw
        self.target_found     = False
        self.target_cx        = 0.0
        self.target_cy        = 0.0
        self.target_area_val  = 0.0
        self.target_id        = -1

        self.trailer_found    = False
        self.trailer_cx       = 0.0
        self.trailer_cy       = 0.0
        self.trailer_area_val = 0.0
        self.trailer_id       = -1

        # Detection data — filtered (used for control)
        self.target_cx_f   = 0.0
        self.target_area_f = 0.0
        self.trailer_cx_f  = 0.0
        self.trailer_area_f= 0.0

        # Data freshness — track last time we received detection data
        self._last_data_time = time.time()
        self._data_timeout   = 2.5  # seconds without data → treat as lost

        # Subscribers
        self.create_subscription(Bool,              'aruco/target/found',          self._cb_target_found,      10)
        self.create_subscription(Float32MultiArray, 'aruco/target/data',           self._cb_target_data,       10)
        self.create_subscription(Float32MultiArray, 'aruco/target/data_filtered',  self._cb_target_filt,       10)
        self.create_subscription(Bool,              'aruco/trailer/found',         self._cb_trailer_found,     10)
        self.create_subscription(Float32MultiArray, 'aruco/trailer/data',          self._cb_trailer_data,      10)
        self.create_subscription(Float32MultiArray, 'aruco/trailer/data_filtered', self._cb_trailer_filt,      10)

        self.state_pub = self.create_publisher(String, 'robot/state', 10)
        self.timer     = self.create_timer(0.1, self.control_loop)

    # ── Callbacks ────────────────────────────────────────────────────
    def _cb_target_found(self,  m): self.target_found  = m.data
    def _cb_trailer_found(self, m): self.trailer_found = m.data

    def _cb_target_data(self, m):
        self.target_cx, self.target_cy, self.target_area_val, id_ = m.data
        self.target_id    = int(id_)
        self.target_found = self.target_id != -1
        self._last_data_time = time.time()

    def _cb_target_filt(self, m):
        self.target_cx_f, _, self.target_area_f, _ = m.data

    def _cb_trailer_data(self, m):
        self.trailer_cx, self.trailer_cy, self.trailer_area_val, id_ = m.data
        self.trailer_id    = int(id_)
        self.trailer_found = self.trailer_id != -1
        self._last_data_time = time.time()

    def _cb_trailer_filt(self, m):
        self.trailer_cx_f, _, self.trailer_area_f, _ = m.data

    # ── Properties — same pattern as original ────────────────────────
    @property
    def last_found(self):
        return self.trailer_found if self.searching_container else self.target_found

    @property
    def last_cx(self):
        """Filtered cx for control."""
        return self.trailer_cx_f if self.searching_container else self.target_cx_f

    @property
    def last_area(self):
        """Filtered area for control."""
        return self.trailer_area_f if self.searching_container else self.target_area_f

    @property
    def last_id(self):
        return self.trailer_id if self.searching_container else self.target_id

    @property
    def data_fresh(self):
        """True if we received detection data recently."""
        return (time.time() - self._last_data_time) < self._data_timeout

    # ── Motion helpers — identical to original ────────────────────────
    def start_motion(self, duration):
        self.motion_done_time = self.get_clock().now() + Duration(seconds=duration)
        self.motion_active    = True

    def motion_done(self):
        if not self.motion_active:
            return True
        if self.get_clock().now() >= self.motion_done_time:
            self.motion_active = False
            return True
        return False

    def _xgo(self, method, *args):
        if self.xgo:
            getattr(self.xgo, method)(*args)

    # ── Control loop — same structure as original ─────────────────────
    def control_loop(self):
        msg = String(); msg.data = self.state.name
        self.state_pub.publish(msg)

        if self.motion_active:
            if not self.motion_done():
                return

        if self.state != self.prev_state:
            self.get_logger().info(f"[FSM] {self.prev_state} → {self.state}")
            self.prev_state = self.state

        # =========================
        # SEARCH
        # =========================
        if self.state == State.SEARCH:
            if self.last_found:
                self.state = State.ALIGN
                return
            self._xgo('turn', 25)
            self.start_motion(1.0)
            self.state = State.SEARCH_TURNING

        elif self.state == State.SEARCH_TURNING:
            if self.motion_done():
                self._xgo('stop')
                self.start_motion(2.0)
                self.state = State.SEARCH_WAIT

        elif self.state == State.SEARCH_WAIT:
            if self.motion_done():
                self.state = State.SEARCH

        # =========================
        # ALIGN
        # =========================
        elif self.state == State.ALIGN:
            if not self.last_found:
                self.state = State.SEARCH
                return

            cx_error = (self.last_cx - self.image_width / 2.0) / (self.image_width / 2.0)

            if abs(cx_error) > self.cx_threshold:
                if cx_error > 0:
                    turn   = min((-self.turn_gain * cx_error + 20.0), -(self.min_turn + 20.0))
                    strave = min((-self.strave_gain * cx_error), -(self.min_strave))
                else:
                    turn   = max((-self.turn_gain * cx_error), self.min_turn)
                    strave = max((-self.strave_gain * cx_error), self.min_strave)

                self.get_logger().info(f"[FSM] cx_error={cx_error:.3f}  turn={turn:.1f}  strave={strave:.1f}")

                if abs(cx_error) > self.cx_threshold * 2.0:
                    self._xgo('turn', turn)
                else:
                    self._xgo('move', "y", strave)

                self.start_motion(1.0)
                self.state = State.ALIGN_TURNING
            else:
                self.state = State.APPROACH

        elif self.state == State.ALIGN_TURNING:
            if self.motion_done():
                self._xgo('stop')
                self.start_motion(2.0)
                self.state = State.ALIGN_WAIT

        elif self.state == State.ALIGN_WAIT:
            if self.motion_done():
                self.state = State.ALIGN

        # =========================
        # APPROACH
        # =========================
        elif self.state == State.APPROACH:
            # If data stream dropped, stop and go back to search
            if not self.data_fresh:
                self.get_logger().warn("[FSM] Detection data stale — backing off to SEARCH")
                self._xgo('stop')
                self.start_motion(1.0)
                self.state = State.APPROACH_BACK
                return
            if not self.last_found:
                self._xgo('move', "x", -self.forward_speed)
                self.start_motion(1.0)
                self.state = State.APPROACH_BACK
                return

            target = self.container_target_area if self.searching_container else self.target_area
            self.get_logger().info(f"[FSM] target_area={target:.0f}  actual={self.last_area:.0f}")

            if self.last_area < target:
                error    = target - self.last_area
                duration = max(error / target * self.forward_gain, 0.2) if (error / target > 0.90) else 1.0
                self.get_logger().info(f"[FSM] area_error={error:.0f}  duration={duration:.2f}")
                self._xgo('move', "x", self.forward_speed)
                self.start_motion(duration)
                self.state = State.APPROACH_FORWARD
            else:
                # Both target and trailer go directly to their next state.
                # Strafe is handled as substeps inside DEPLOY.
                self.state = State.DEPLOY if self.searching_container else State.GRASP

        elif self.state == State.APPROACH_FORWARD:
            if self.motion_done():
                self._xgo('stop')
                self.start_motion(2.0)
                self.state = State.APPROACH

        elif self.state == State.APPROACH_BACK:
            if self.motion_done():
                self._xgo('stop')
                self.start_motion(2.0)
                if self.forward_speed > 7.5 and not self.searching_container:
                    self.forward_speed *= 0.75
                self.state = State.ALIGN

        # =========================
        # GRASP
        # =========================
        elif self.state == State.GRASP:
            if self.substep == 0:
                self._xgo('claw', 0)
                self.start_motion(1.0)
                self.substep = 1

            elif self.substep == 1 and self.motion_done():
                self._xgo('arm_motor', [-25, 90, 0])
                self.start_motion(1.0)
                self.substep = 2

            elif self.substep == 2 and self.motion_done():
                self._xgo('claw', 255)
                self.start_motion(2.0)
                self.substep = 3

            elif self.substep == 3 and self.motion_done():
                self._xgo('arm_motor', [20, -90, 0])
                self.start_motion(1.0)
                self.substep = 4

            elif self.substep == 4 and self.motion_done():
                self._xgo('arm_motor', [83, -90, 0])
                self.start_motion(1.0)
                self.substep = 5

            elif self.substep == 5 and self.motion_done():
                self.substep      = 0
                self.picked_up_id = self.last_id
                self.state        = State.VERIFY

        # =========================
        # VERIFY
        # =========================
        elif self.state == State.VERIFY:
            if self.substep == 0:
                self._xgo('stop')
                self.start_motion(3.0)
                self.substep = 1

            elif self.substep == 1 and self.motion_done():
                self._xgo('move', "x", -self.forward_speed)
                self.start_motion(1.0)
                self.substep = 2

            elif self.substep == 2 and self.motion_done():
                self._xgo('stop')
                self.start_motion(3.0)
                self.substep = 3

            elif self.substep == 3 and self.motion_done():
                verified = (self.target_id == -1)
                self.get_logger().info(f"[FSM] Verified: {verified}")
                self.substep = 0

                if verified:
                    self._xgo('translation', "z", -50)
                    self._xgo('attitude',    "p",  -10)
                    self.start_motion(2.0)
                    self.forward_speed       = self.get_parameter('forward_speed').value
                    self.searching_container = True
                    self.state               = State.SEARCH
                else:
                    self._xgo('translation', "z", 80)
                    self._xgo('attitude',    "p", 15)
                    self.start_motion(2.0)
                    self.forward_speed = self.get_parameter('forward_speed').value
                    self.state         = State.SEARCH

        # =========================
        # DEPLOY
        # Substeps 0-1: strafe left to align with box opening
        # Substeps 2-6: original deploy arm sequence
        # =========================
        elif self.state == State.DEPLOY:
            # ── Strafe substeps ──────────────────────────────────────
            if self.substep == 0:
                self._xgo('move', "y", self.strafe_left_speed)  # positive y = left
                self.start_motion(self.strafe_left_duration)
                self.substep = 1

            elif self.substep == 1 and self.motion_done():
                self._xgo('stop')
                self.start_motion(0.5)
                self.substep = 2

            # ── Original deploy sequence (shifted by 2) ───────────────
            elif self.substep == 2 and self.motion_done():
                self._xgo('arm_motor', [-270, 210, 0])
                self.start_motion(1.0)
                self.substep = 3

            elif self.substep == 3 and self.motion_done():
                self._xgo('claw', 0)
                self.start_motion(1.0)
                self.substep = 4

            elif self.substep == 4 and self.motion_done():
                self._xgo('arm_motor', [0, -90, 0])
                self.start_motion(1.0)
                self.substep = 5

            elif self.substep == 5 and self.motion_done():
                self._xgo('arm_motor', [83, -90, 0])
                self.start_motion(1.0)
                self.substep = 6

            elif self.substep == 6 and self.motion_done():
                self._xgo('translation', "z", 0)
                self._xgo('attitude',    "p", 0)
                self.start_motion(2.0)
                self.substep = 7

            elif self.substep == 7 and self.motion_done():
                self.substep = 0
                self.state   = State.DONE

        # =========================
        # DONE
        # =========================
        elif self.state == State.DONE:
            self._xgo('action', 15)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.xgo:
            node.xgo.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
