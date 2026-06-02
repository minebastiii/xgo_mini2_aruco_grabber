#!/usr/bin/env python3
"""
fsm_node.py

Steuerungslogik:  aruco_fsm.py (Original, komplett unverändert)
Schnittstelle:    neue Laptop-Node (detector_node.py)

Änderungen gegenüber aruco_fsm.py:
  1. Neue Topics der Laptop-Node statt aruco/data
  2. robot/state Publisher
  3. Distanz- oder Area-basierter GRASP/DEPLOY-Trigger je nach Modus:
     - use_distance_target  : ob Target-Phase Distanz oder Area nutzt
     - use_distance_trailer : ob Trailer-Phase Distanz oder Area nutzt
     - Umschaltbar zur Laufzeit über Topic fsm/control (JSON String)
     - Aktueller Modus wird auf fsm/control_mode publiziert
  WICHTIG: Alle Timings, Substeps, Wartezeiten, Bewegungsbefehle
           sind 100% identisch zum Original aruco_fsm.py
"""

import json
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool, Float32MultiArray, Float32, String
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

        # ── Parameter (identisch zum Original) ───────────────────────
        self.declare_parameter('target_marker_area',    22500.0)
        self.declare_parameter('container_marker_area', 60000.0)
        self.declare_parameter('turn_gain',             30.0)
        self.declare_parameter('forward_gain',          3.75)
        self.declare_parameter('forward_speed',         10.0)
        self.declare_parameter('cx_threshold',          0.15)
        self.declare_parameter('min_turn',              7.0)
        self.declare_parameter('grasp_distance_m',      0.15)
        self.declare_parameter('deploy_distance_m',     0.15)
        # Startmodus: True = Distanz, False = Area
        self.declare_parameter('use_distance_target',   True)
        self.declare_parameter('use_distance_trailer',  True)

        self.target_area           = self.get_parameter('target_marker_area').value
        self.container_target_area = self.get_parameter('container_marker_area').value
        self.turn_gain             = self.get_parameter('turn_gain').value
        self.forward_gain          = self.get_parameter('forward_gain').value
        self.forward_speed         = self.get_parameter('forward_speed').value
        self.cx_threshold          = self.get_parameter('cx_threshold').value
        self.min_turn              = self.get_parameter('min_turn').value
        self.grasp_distance_m      = self.get_parameter('grasp_distance_m').value
        self.deploy_distance_m     = self.get_parameter('deploy_distance_m').value
        self.min_strave            = 5.0
        self.strave_gain           = 10.0

        # Laufzeit-Modus (umschaltbar über fsm/control Topic)
        self.use_distance_target  = self.get_parameter('use_distance_target').value
        self.use_distance_trailer = self.get_parameter('use_distance_trailer').value

        self.once = True

        # ── XGO ──────────────────────────────────────────────────────
        if XGO_AVAILABLE:
            self.xgo = XGO(port='/dev/ttyAMA0')
            self.xgo.stop()
            self.xgo.translation("z", 80)
            self.xgo.attitude("p", 15)
        else:
            self.xgo = None
            self.get_logger().warn("XGO not available — Motorbefehle unterdrückt")

        # ── FSM-Zustand ───────────────────────────────────────────────
        self.state               = State.SEARCH
        self.prev_state          = None
        self.motion_active       = False
        self.substep             = 0
        self.searching_container = False
        self.picked_up_id        = -1
        self.get_logger().info("[FSM] Starting new FSM")

        # ── Detektionsdaten ───────────────────────────────────────────
        self.last_found = False
        self.last_cx    = 0.0
        self.last_cy    = 0.0
        self.last_area  = 0.0
        self.last_id    = -1

        self._target_cx      = 0.0
        self._target_cy      = 0.0
        self._target_area    = 0.0
        self._target_id      = -1
        self._target_cx_f    = 0.0
        self._target_area_f  = 0.0
        self._trailer_cx     = 0.0
        self._trailer_cy     = 0.0
        self._trailer_area   = 0.0
        self._trailer_id     = -1
        self._trailer_cx_f   = 0.0
        self._trailer_area_f = 0.0
        self._target_distance  = -1.0
        self._trailer_distance = -1.0

        # ── Subscriber ────────────────────────────────────────────────
        self.create_subscription(Float32MultiArray, 'aruco/target/data',
                                 self._cb_target_data, 10)
        self.create_subscription(Float32MultiArray, 'aruco/target/data_filtered',
                                 self._cb_target_filt, 10)
        self.create_subscription(Float32MultiArray, 'aruco/trailer/data',
                                 self._cb_trailer_data, 10)
        self.create_subscription(Float32MultiArray, 'aruco/trailer/data_filtered',
                                 self._cb_trailer_filt, 10)
        self.create_subscription(Float32, 'aruco/target/distance',
                                 self._cb_target_distance, 10)
        self.create_subscription(Float32, 'aruco/trailer/distance',
                                 self._cb_trailer_distance, 10)
        # Dashboard → FSM: Modus umschalten
        self.create_subscription(String, 'fsm/control',
                                 self._cb_control, 10)

        # ── Publisher ─────────────────────────────────────────────────
        self.state_pub   = self.create_publisher(String, 'robot/state',       10)
        self.mode_pub    = self.create_publisher(String, 'fsm/control_mode',  10)

        self.timer = self.create_timer(0.1, self.control_loop)

    # ── Callbacks ────────────────────────────────────────────────────

    def _cb_target_data(self, msg: Float32MultiArray):
        self._target_cx, self._target_cy, self._target_area, id_ = msg.data
        self._target_id = int(id_)
        if not self.searching_container:
            self.last_cx    = self._target_cx
            self.last_cy    = self._target_cy
            self.last_area  = self._target_area
            self.last_id    = self._target_id
            self.last_found = self.last_id != -1

    def _cb_target_filt(self, msg: Float32MultiArray):
        self._target_cx_f, _, self._target_area_f, _ = msg.data
        if not self.searching_container:
            self.last_cx   = self._target_cx_f
            self.last_area = self._target_area_f

    def _cb_trailer_data(self, msg: Float32MultiArray):
        self._trailer_cx, self._trailer_cy, self._trailer_area, id_ = msg.data
        self._trailer_id = int(id_)
        if self.searching_container:
            self.last_cx    = self._trailer_cx
            self.last_cy    = self._trailer_cy
            self.last_area  = self._trailer_area
            self.last_id    = self._trailer_id
            self.last_found = self.last_id != -1

    def _cb_trailer_filt(self, msg: Float32MultiArray):
        self._trailer_cx_f, _, self._trailer_area_f, _ = msg.data
        if self.searching_container:
            self.last_cx   = self._trailer_cx_f
            self.last_area = self._trailer_area_f

    def _cb_target_distance(self,  msg: Float32): self._target_distance  = float(msg.data)
    def _cb_trailer_distance(self, msg: Float32): self._trailer_distance = float(msg.data)

    def _cb_control(self, msg: String):
        """
        Empfängt JSON vom Dashboard, z.B.:
          {"use_distance_target": true}
          {"use_distance_trailer": false}
          {"grasp_distance_m": 0.18}
          {"deploy_distance_m": 0.22}
        """
        try:
            cmd = json.loads(msg.data)
            if "use_distance_target" in cmd:
                self.use_distance_target  = bool(cmd["use_distance_target"])
                self.get_logger().info(
                    f"[FSM] Target mode → {'distance' if self.use_distance_target else 'area'}")
            if "use_distance_trailer" in cmd:
                self.use_distance_trailer = bool(cmd["use_distance_trailer"])
                self.get_logger().info(
                    f"[FSM] Trailer mode → {'distance' if self.use_distance_trailer else 'area'}")
            if "grasp_distance_m" in cmd:
                self.grasp_distance_m  = float(cmd["grasp_distance_m"])
            if "deploy_distance_m" in cmd:
                self.deploy_distance_m = float(cmd["deploy_distance_m"])
        except Exception as e:
            self.get_logger().error(f"[FSM] control parse error: {e}")

    # ── Motion-Hilfsfunktionen ────────────────────────────────────────

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

    # ── Trigger-Hilfsfunktionen ───────────────────────────────────────

    def _target_trigger(self):
        """True wenn der GRASP-Trigger für die Target-Phase ausgelöst werden soll."""
        if self.use_distance_target:
            result = self._target_distance > 0.0 and self._target_distance <= self.grasp_distance_m
            self.get_logger().info(
                f"[FSM] _target_trigger [distance] "
                f"dist={self._target_distance:.3f}m  threshold={self.grasp_distance_m}m  → {result}")
            return result
        else:
            result = self.last_area >= self.target_area
            self.get_logger().info(
                f"[FSM] _target_trigger [area] "
                f"area={self.last_area:.0f}  threshold={self.target_area:.0f}  → {result}")
            return result

    def _trailer_trigger(self):
        """True wenn der DEPLOY-Trigger für die Trailer-Phase ausgelöst werden soll."""
        if self.use_distance_trailer:
            result = self._trailer_distance > 0.0 and self._trailer_distance <= self.deploy_distance_m
            self.get_logger().info(
                f"[FSM] _trailer_trigger [distance] "
                f"dist={self._trailer_distance:.3f}m  threshold={self.deploy_distance_m}m  → {result}")
            return result
        else:
            result = self.last_area >= self.container_target_area
            self.get_logger().info(
                f"[FSM] _trailer_trigger [area] "
                f"area={self.last_area:.0f}  threshold={self.container_target_area:.0f}  → {result}")
            return result

    # ── Control-Loop ─────────────────────────────────────────────────
    def control_loop(self):
        # State publizieren
        s_msg = String(); s_msg.data = self.state.name
        self.state_pub.publish(s_msg)

        # Modus publizieren (für Dashboard)
        mode = {
            "use_distance_target":  self.use_distance_target,
            "use_distance_trailer": self.use_distance_trailer,
            "grasp_distance_m":     self.grasp_distance_m,
            "deploy_distance_m":    self.deploy_distance_m,
            "target_distance":      self._target_distance,
            "trailer_distance":     self._trailer_distance,
        }
        m_msg = String(); m_msg.data = json.dumps(mode)
        self.mode_pub.publish(m_msg)

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
            self.xgo.turn(25)
            self.start_motion(1.0)
            self.state = State.SEARCH_TURNING

        elif self.state == State.SEARCH_TURNING:
            if self.motion_done():
                self.xgo.stop()
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

            cx_error = (self.last_cx - 960.0) / 960.0

            if abs(cx_error) > self.cx_threshold:
                if cx_error > 0:
                    turn   = min((-self.turn_gain * cx_error + 20.0), -(self.min_turn + 20.0))
                    strave = min((-self.strave_gain * cx_error), -(self.min_strave))
                else:
                    turn   = max((-self.turn_gain * cx_error), self.min_turn)
                    strave = max((-self.strave_gain * cx_error), self.min_strave)

                self.get_logger().info(f"[FSM] Error : {cx_error},  Turning : {turn}")
                self.get_logger().info(f"[FSM] Error : {cx_error},  Straving : {strave}")

                if abs(cx_error) > self.cx_threshold * 2.0:
                    self.xgo.turn(turn)
                else:
                    self.xgo.move("y", strave)

                self.start_motion(1.0)
                self.state = State.ALIGN_TURNING
            else:
                self.state = State.APPROACH

        elif self.state == State.ALIGN_TURNING:
            if self.motion_done():
                self.xgo.stop()
                self.start_motion(2.0)
                self.state = State.ALIGN_WAIT

        elif self.state == State.ALIGN_WAIT:
            if self.motion_done():
                self.state = State.ALIGN

        # =========================
        # APPROACH
        # =========================
        elif self.state == State.APPROACH:
            if not self.last_found:
                self.xgo.move("x", -self.forward_speed)
                self.start_motion(1.0)
                self.state = State.APPROACH_BACK
                return

            # Trigger prüfen — NUR der konfigurierte Modus (Distanz ODER Area), kein Fallback
            if not self.searching_container:
                if self._target_trigger():
                    self.get_logger().info("[FSM] Target trigger reached → GRASP")
                    self.state = State.GRASP
                    return
            else:
                if self._trailer_trigger():
                    self.get_logger().info("[FSM] Trailer trigger reached → DEPLOY")
                    self.state = State.DEPLOY
                    return

            # Noch nicht nah genug — vorwärts fahren (originale Logik unverändert)
            target = self.container_target_area if self.searching_container else self.target_area
            self.get_logger().info(f"[FSM] Target area : {target} Actual area : {self.last_area}")

            if self.last_area < target:
                error = target - self.last_area
                if (error / target > 0.90):
                    duration = max(error / target * self.forward_gain, 0.2)
                else:
                    duration = 1.0
                self.get_logger().info(
                    f"[FSM] area error={error:.2f} error/target={error / target:.2f} duration={duration:.2f}"
                )
                self.xgo.move("x", self.forward_speed)
                self.start_motion(duration)
                self.state = State.APPROACH_FORWARD

        elif self.state == State.APPROACH_FORWARD:
            if self.motion_done():
                self.xgo.stop()
                self.start_motion(2.0)
                self.state = State.APPROACH

        elif self.state == State.APPROACH_BACK:
            if self.motion_done():
                self.xgo.stop()
                self.start_motion(2.0)
                if self.forward_speed > 7.5 and not self.searching_container:
                    self.forward_speed *= 0.75
                self.state = State.ALIGN

        # =========================
        # GRASP  (identisch zum Original)
        # =========================
        elif self.state == State.GRASP:
            if self.substep == 0:
                self.xgo.claw(0)
                self.start_motion(1.0)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.move("x", 2.5)
                self.start_motion(1.0)
                self.substep = 2
            elif self.substep == 2 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                self.xgo.arm_motor([-25, 90, 0])
                self.start_motion(1.0)
                self.substep = 4
            elif self.substep == 4 and self.motion_done():
                self.xgo.claw(255)
                self.start_motion(2.0)
                self.substep = 5
            elif self.substep == 5 and self.motion_done():
                self.xgo.arm_motor([20, -90, 0])
                self.start_motion(1.0)
                self.substep = 6
            elif self.substep == 6 and self.motion_done():
                self.xgo.arm_motor([83, -90, 0])
                self.start_motion(1.0)
                self.substep = 7
            elif self.substep == 7 and self.motion_done():
                self.substep      = 0
                self.picked_up_id = self.last_id
                self.last_id      = -1
                self.state        = State.VERIFY

        # =========================
        # VERIFY  (identisch zum Original)
        # =========================
        elif self.state == State.VERIFY:
            if self.substep == 0:
                self.xgo.stop()
                self.start_motion(3.0)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.move("x", -self.forward_speed)
                self.start_motion(1.0)
                self.substep = 2
            elif self.substep == 2:
                # kein motion_done()-Check — identisch zum Original
                self.xgo.stop()
                self.start_motion(3.0)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                verified = True if self.last_id == -1 else False
                self.get_logger().info(f"Verified : {verified}")
                self.substep = 0
                if verified:
                    self.xgo.translation("z", 0) #-50)
                    self.xgo.attitude("p", 0) #-10)
                    self.start_motion(2.0)
                    self.forward_speed       = self.get_parameter('forward_speed').value
                    self.last_found          = False
                    self.searching_container = True
                    self.state               = State.SEARCH
                else:
                    self.xgo.translation("z", 80)
                    self.xgo.attitude("p", 15)
                    self.start_motion(2.0)
                    self.last_found    = False
                    self.forward_speed = self.get_parameter('forward_speed').value
                    self.state         = State.SEARCH

        # =========================
        # DEPLOY  (identisch zum Original)
        # =========================
        elif self.state == State.DEPLOY:
            # if self.once:
            #     self.xgo.translation("z", 0)
            #     self.xgo.attitude("p", 0)
            #     self.start_motion(2.0)
            #     self.once = False

            if self.substep == 0 and self.motion_done():
                self.xgo.move("y", 5)
                self.start_motion(2.0)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 2
            elif self.substep == 2:
                self.xgo.arm(130, -20)
                #self.xgo.arm_motor([-270, 210, 0])
                #self.xgo.arm_motor([-270, 210, 0])
                self.start_motion(1.0)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                self.xgo.claw(0)
                self.start_motion(1.0)
                self.substep = 4
            elif self.substep == 4 and self.motion_done():
                self.xgo.arm_motor([0, -90, 0])
                self.start_motion(1.0)
                self.substep = 5
            elif self.substep == 5 and self.motion_done():
                self.xgo.arm_motor([83, -90, 0])
                self.start_motion(1.0)
                self.substep = 6
            elif self.substep == 6 and self.motion_done():
                self.xgo.translation("z", 0)
                self.xgo.attitude("p", 0)
                self.start_motion(2.0)
                self.substep = 7
            elif self.substep == 7 and self.motion_done():
                self.substep = 0
                self.state   = State.DONE

        # =========================
        # DONE  (identisch zum Original)
        # =========================
        elif self.state == State.DONE:
            self.xgo.action(15)


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
