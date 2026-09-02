#!/usr/bin/env python3
"""
fsm_pose_node.py

Steuerungslogik:  aruco_fsm.py (Original), erweitert um Posen-Annäherung
Schnittstelle:    camera_pose_node.py (Target- UND Trailer-Pose)

Änderungen gegenüber dem Original:
  1. Neue Topics der camera_pose_node statt aruco/data
  2. robot/state Publisher
  3. Distanz- oder Area-basierter GRASP/DEPLOY-Trigger je nach Modus
     (use_distance_target / use_distance_trailer, umschaltbar zur
     Laufzeit über fsm/control)
  4. "Square-up" vor dem eigentlichen Vorwärtsfahren — regelt den
     seitlichen Versatz (mx) zur Markernormale auf ~0, bevor
     APPROACH_FORWARD losläuft. Gilt für Target UND Trailer, gesteuert
     über die gleichen States (APPROACH_SQUARE / APPROACH_SQUARE_WAIT).
  5. NEU: Debounce für "Marker nicht gefunden" — APPROACH_BACK wird
     erst ausgelöst, wenn der Marker mehrere Zyklen in Folge (nicht nur
     einen einzelnen Frame) nicht erkannt wurde. Verhindert, dass kurze
     Detection-Aussetzer (Kompressionsartefakte, Blur, Bildrand) den
     Roboter unnötig zurückfahren lassen, statt weiter auf das Ziel
     zuzulaufen.
  WICHTIG: Alle Timings, Substeps, Wartezeiten, Bewegungsbefehle in
           GRASP/VERIFY/DEPLOY/DONE sind 100% identisch zum Original.
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
    SEARCH                = auto()
    SEARCH_TURNING        = auto()
    SEARCH_WAIT           = auto()
    ALIGN                 = auto()
    ALIGN_TURNING         = auto()
    ALIGN_WAIT            = auto()
    APPROACH               = auto()
    APPROACH_SQUARE        = auto()   # seitlichen Versatz ausregeln
    APPROACH_SQUARE_WAIT   = auto()
    APPROACH_FORWARD       = auto()
    APPROACH_BACK           = auto()
    GRASP                  = auto()
    VERIFY                  = auto()
    DEPLOY                  = auto()
    DONE                    = auto()


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
        # ── Distanz-Modus Annäherungsregelung ────────────────────────
        self.declare_parameter('approach_dist_gain', 10.0)   # s/m
        self.declare_parameter('approach_step_min',  1.0)    # s
        self.declare_parameter('approach_step_max',  3.0)    # s
        # ── Seitlicher Versatz (Square-up) ────────────────────────────
        self.declare_parameter('lateral_tolerance_m', 0.03)  # m, ab wann "genau davor"
        self.declare_parameter('lateral_gain',        8.0)   # s/m
        self.declare_parameter('lateral_step_min',    0.3)   # s
        self.declare_parameter('lateral_step_max',    1.5)   # s
        self.declare_parameter('lateral_speed',       8.0)   # XGO move("y", ...) Einheit
        self.declare_parameter('lateral_sign',        1.0)   # +1/-1, falls Strafe falschrum
        # ── NEU: Debounce für "Marker nicht gefunden" ─────────────────
        self.declare_parameter('not_found_debounce', 3)  # Anzahl aufeinanderfolgender Miss-Frames

        self.target_area           = self.get_parameter('target_marker_area').value
        self.container_target_area = self.get_parameter('container_marker_area').value
        self.turn_gain             = self.get_parameter('turn_gain').value
        self.forward_gain          = self.get_parameter('forward_gain').value
        self.forward_speed         = self.get_parameter('forward_speed').value
        self.cx_threshold          = self.get_parameter('cx_threshold').value
        self.min_turn               = self.get_parameter('min_turn').value
        self.grasp_distance_m       = self.get_parameter('grasp_distance_m').value
        self.deploy_distance_m      = self.get_parameter('deploy_distance_m').value
        self.min_strave              = 5.0
        self.strave_gain             = 10.0
        self.approach_dist_gain     = self.get_parameter('approach_dist_gain').value
        self.approach_step_min      = self.get_parameter('approach_step_min').value
        self.approach_step_max      = self.get_parameter('approach_step_max').value

        self.lateral_tolerance_m = self.get_parameter('lateral_tolerance_m').value
        self.lateral_gain        = self.get_parameter('lateral_gain').value
        self.lateral_step_min    = self.get_parameter('lateral_step_min').value
        self.lateral_step_max    = self.get_parameter('lateral_step_max').value
        self.lateral_speed       = self.get_parameter('lateral_speed').value
        self.lateral_sign        = self.get_parameter('lateral_sign').value

        self.not_found_debounce  = self.get_parameter('not_found_debounce').value

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

        # Pose-Daten (aus camera_pose_node: aruco/<target|trailer>/pose)
        self._target_distance  = -1.0   # mz
        self._target_lateral   = 0.0    # mx
        self._target_bearing   = 0.0
        self._trailer_distance = -1.0   # mz
        self._trailer_lateral  = 0.0    # mx
        self._trailer_bearing  = 0.0

        # NEU: Streak-Zähler für aufeinanderfolgende Nicht-Erkennungen
        self._target_not_found_streak  = 0
        self._trailer_not_found_streak = 0

        # ── Subscriber ────────────────────────────────────────────────
        self.create_subscription(Float32MultiArray, 'aruco/target/data',
                                 self._cb_target_data, 10)
        self.create_subscription(Float32MultiArray, 'aruco/target/data_filtered',
                                 self._cb_target_filt, 10)
        self.create_subscription(Float32MultiArray, 'aruco/trailer/data',
                                 self._cb_trailer_data, 10)
        self.create_subscription(Float32MultiArray, 'aruco/trailer/data_filtered',
                                 self._cb_trailer_filt, 10)
        # Pose (mx, mz, bearing, id)
        self.create_subscription(Float32MultiArray, 'aruco/target/pose',
                                 self._cb_target_pose, 10)
        self.create_subscription(Float32MultiArray, 'aruco/trailer/pose',
                                 self._cb_trailer_pose, 10)
        # Dashboard → FSM: Modus umschalten
        self.create_subscription(String, 'fsm/control',
                                 self._cb_control, 10)

        # ── Publisher ─────────────────────────────────────────────────
        self.state_pub   = self.create_publisher(String, 'robot/state',       10)
        self.mode_pub    = self.create_publisher(String, 'fsm/control_mode',  10)
        self.battery_pub = self.create_publisher(Float32, 'robot/battery',    10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.create_timer(10.0, self._pub_battery)
        self._pub_battery()

    # ── Callbacks ────────────────────────────────────────────────────

    def _cb_target_data(self, msg: Float32MultiArray):
        self._target_cx, self._target_cy, self._target_area, id_ = msg.data
        self._target_id = int(id_)

        if self._target_id == -1:
            self._target_not_found_streak += 1
        else:
            self._target_not_found_streak = 0

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

        if self._trailer_id == -1:
            self._trailer_not_found_streak += 1
        else:
            self._trailer_not_found_streak = 0

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

    def _cb_target_pose(self, msg: Float32MultiArray):
        mx, mz, bearing_deg, id_ = msg.data
        self._target_lateral  = mx
        self._target_bearing  = bearing_deg
        self._target_distance = mz if int(id_) != -1 else -1.0

    def _cb_trailer_pose(self, msg: Float32MultiArray):
        mx, mz, bearing_deg, id_ = msg.data
        self._trailer_lateral  = mx
        self._trailer_bearing  = bearing_deg
        self._trailer_distance = mz if int(id_) != -1 else -1.0

    def _cb_control(self, msg: String):
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
            if "approach_dist_gain" in cmd:
                self.approach_dist_gain = float(cmd["approach_dist_gain"])
                self.get_logger().info(f"[FSM] approach_dist_gain → {self.approach_dist_gain}")
            if "approach_step_min" in cmd:
                self.approach_step_min = float(cmd["approach_step_min"])
            if "approach_step_max" in cmd:
                self.approach_step_max = float(cmd["approach_step_max"])
            if "lateral_tolerance_m" in cmd:
                self.lateral_tolerance_m = float(cmd["lateral_tolerance_m"])
            if "lateral_gain" in cmd:
                self.lateral_gain = float(cmd["lateral_gain"])
            if "lateral_sign" in cmd:
                self.lateral_sign = float(cmd["lateral_sign"])
            if "not_found_debounce" in cmd:
                self.not_found_debounce = int(cmd["not_found_debounce"])
                self.get_logger().info(f"[FSM] not_found_debounce → {self.not_found_debounce}")
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

    # ── Trigger-/Pose-Hilfsfunktionen ─────────────────────────────────

    def _target_trigger(self):
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

    def _current_lateral_mode(self):
        """Gibt (lateral_mx, use_distance_mode) für den GERADE aktiven
        Marker zurück (Target oder Trailer, je nach searching_container).
        Squaring läuft nur im Distanz/Pose-Modus, da im reinen Area-Modus
        keine mx-Schätzung vorliegt."""
        if not self.searching_container:
            return self._target_lateral, self.use_distance_target
        else:
            return self._trailer_lateral, self.use_distance_trailer

    def _current_not_found_streak(self):
        """Anzahl aufeinanderfolgender Miss-Frames für den gerade
        aktiven Marker (Target oder Trailer)."""
        return self._trailer_not_found_streak if self.searching_container else self._target_not_found_streak

    # ── Battery ───────────────────────────────────────────────────────
    def _pub_battery(self):
        if self.xgo is None:
            return
        try:
            level = self.xgo.read_battery()
            msg = Float32()
            msg.data = float(level)
            self.battery_pub.publish(msg)
            if level <= 20:
                self.get_logger().warn(f"[FSM] Battery LOW: {level}%")
            else:
                self.get_logger().debug(f"[FSM] Battery: {level}%")
        except Exception as e:
            self.get_logger().warn(f"[FSM] Battery read failed: {e}")

    # ── Control-Loop ─────────────────────────────────────────────────
    def control_loop(self):
        s_msg = String(); s_msg.data = self.state.name
        self.state_pub.publish(s_msg)

        mode = {
            "use_distance_target":  self.use_distance_target,
            "use_distance_trailer": self.use_distance_trailer,
            "grasp_distance_m":     self.grasp_distance_m,
            "deploy_distance_m":    self.deploy_distance_m,
            "target_distance":      self._target_distance,
            "target_lateral":       self._target_lateral,
            "trailer_distance":     self._trailer_distance,
            "trailer_lateral":      self._trailer_lateral,
            "approach_dist_gain":   self.approach_dist_gain,
            "approach_step_min":    self.approach_step_min,
            "approach_step_max":    self.approach_step_max,
            "lateral_tolerance_m":  self.lateral_tolerance_m,
            "lateral_gain":         self.lateral_gain,
            "not_found_debounce":   self.not_found_debounce,
            "target_not_found_streak":  self._target_not_found_streak,
            "trailer_not_found_streak": self._trailer_not_found_streak,
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
                if self._current_not_found_streak() >= self.not_found_debounce:
                    self.get_logger().warn(
                        f"[FSM] Marker {self.not_found_debounce}x in Folge nicht erkannt "
                        f"→ APPROACH_BACK")
                    self.xgo.move("x", -self.forward_speed)
                    self.start_motion(1.0)
                    self.state = State.APPROACH_BACK
                else:
                    self.get_logger().info(
                        f"[FSM] Marker kurzzeitig verloren "
                        f"({self._current_not_found_streak()}/{self.not_found_debounce}) — warte")
                    self.start_motion(0.1)  # kurz warten statt sofort zurückzufahren
                return

            # ── Trigger prüfen ────────────────────────────────────────
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

            # ── Square-up — seitlichen Versatz (mx) ausregeln ─────────
            lateral, use_dist_mode = self._current_lateral_mode()
            if use_dist_mode and abs(lateral) > self.lateral_tolerance_m:
                direction = 1.0 if lateral > 0 else -1.0
                step_dur = max(self.lateral_step_min,
                                min(self.lateral_step_max,
                                    abs(lateral) * self.lateral_gain))
                strafe = self.lateral_sign * direction * self.lateral_speed
                which = "trailer" if self.searching_container else "target"
                self.get_logger().info(
                    f"[FSM] square-up [{which}]: mx={lateral:.3f}m "
                    f"strafe={strafe:.1f} step={step_dur:.2f}s")
                self.xgo.move("y", strafe)
                self.start_motion(step_dur)
                self.state = State.APPROACH_SQUARE
                return

            # ── Vorwärtsbewegung — strikt modusspezifisch ─────────────
            if not self.searching_container:
                use_dist = self.use_distance_target
                dist_val = self._target_distance
                thresh   = self.grasp_distance_m
                area_val = self.last_area
                area_tgt = self.target_area
            else:
                use_dist = self.use_distance_trailer
                dist_val = self._trailer_distance
                thresh   = self.deploy_distance_m
                area_val = self.last_area
                area_tgt = self.container_target_area

            if use_dist:
                if dist_val > 0:
                    remaining = max(0.0, dist_val - thresh)
                    step_dur  = max(self.approach_step_min,
                                    min(self.approach_step_max,
                                        remaining * self.approach_dist_gain))
                else:
                    step_dur = self.approach_step_min
                self.get_logger().info(
                    f"[FSM] approach [dist] dist={dist_val:.3f}m "
                    f"remaining={max(0.0, dist_val - thresh):.3f}m  step={step_dur:.2f}s")
                self.xgo.move("x", self.forward_speed)
                self.start_motion(step_dur)
                self.state = State.APPROACH_FORWARD

            else:
                self.get_logger().info(
                    f"[FSM] Target area : {area_tgt} Actual area : {area_val}")
                if area_val < area_tgt:
                    error = area_tgt - area_val
                    if (error / area_tgt > 0.90):
                        duration = max(error / area_tgt * self.forward_gain, 0.2)
                    else:
                        duration = 1.0
                    self.get_logger().info(
                        f"[FSM] area error={error:.2f} "
                        f"error/target={error / area_tgt:.2f} duration={duration:.2f}"
                    )
                    self.xgo.move("x", self.forward_speed)
                    self.start_motion(duration)
                    self.state = State.APPROACH_FORWARD

        elif self.state == State.APPROACH_SQUARE:
            if self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.state = State.APPROACH_SQUARE_WAIT

        elif self.state == State.APPROACH_SQUARE_WAIT:
            if self.motion_done():
                # zurück zu ALIGN, weil das Strafen cx wieder verschoben hat
                self.state = State.ALIGN

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
                self.xgo.stop()
                self.start_motion(3.0)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                verified = True if self.last_id == -1 else False
                self.get_logger().info(f"Verified : {verified}")
                self.substep = 0
                if verified:
                    self.xgo.translation("z", 100)
                    self.xgo.attitude("p", 0)
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
            if self.substep == 0 and self.motion_done():
                self.xgo.move("y", 5)
                self.start_motion(1.5)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 2
            if self.substep == 2 and self.motion_done():
                self.xgo.move("x", 5)
                self.start_motion(1.5)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 4
            elif self.substep == 4:
                self.xgo.arm(130, -20)
                self.start_motion(1.0)
                self.substep = 5
            elif self.substep == 5 and self.motion_done():
                self.xgo.claw(0)
                self.start_motion(1.0)
                self.substep = 6
            elif self.substep == 6 and self.motion_done():
                self.xgo.arm_motor([0, -90, 0])
                self.start_motion(1.0)
                self.substep = 7
            elif self.substep == 7 and self.motion_done():
                self.xgo.arm_motor([83, -90, 0])
                self.start_motion(1.0)
                self.substep = 8
            elif self.substep == 8 and self.motion_done():
                self.xgo.translation("z", 0)
                self.xgo.attitude("p", 0)
                self.start_motion(2.0)
                self.substep = 9
            elif self.substep == 9 and self.motion_done():
                self.substep = 0
                self.state   = State.DONE

        # =========================
        # DONE  (identisch zum Original)
        # =========================
        elif self.state == State.DONE:
            if self.substep == 0 and self.motion_done():
                self.xgo.move("x", -10)
                self.start_motion(3.5)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 2
            elif self.substep == 2 and self.motion_done():
                self.xgo.translation("z", 80)
                self.xgo.attitude("p", 15)
                self.start_motion(1.0)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.state               = State.SEARCH
                self.prev_state          = None
                self.motion_active       = False
                self.substep             = 0
                self.searching_container = False
                self.picked_up_id        = -1
                self.get_logger().info("[FSM] Restarted FSM")
                self.substep = 0


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
