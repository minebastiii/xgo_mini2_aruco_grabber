#!/usr/bin/env python3
"""
aruco_fsm — pick-and-place FSM for the XGO.

Input: aruco/markers (String/JSON) from camera_node — list of all
visible target/trailer markers, sorted by area (largest first).

ID lock:
  * In SEARCH (target phase) the FSM locks the first target marker it
    sees (= largest/closest in the frame). From then on ALIGN, APPROACH,
    GRASP and VERIFY only react to that ID; other targets are ignored.
  * After a successful VERIFY the FSM locks the first trailer it sees
    (by group name, so near and far IDs of the same trailer both count).
  * Both locks are released only after DEPLOY has been verified (DONE),
    by the optional lock_timeout_s, or by {"release_lock": true} on
    fsm/control.

VERIFY runs twice per cycle (verify_phase), both times with the same
movement: camera pose down (z=80, pitch=15), step back, stop, check.
  * after GRASP:  locked target no longer visible → carry it, search trailer
                  still visible                   → retry from SEARCH
  * after DEPLOY: locked target no longer visible → DONE (step back,
                                                    release locks, restart)
                  still visible (fell next to the trailer)
                                                  → back to target mode,
                                                    ALIGN on the same ID
"""

import json
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32, String
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

        self.declare_parameter('target_marker_area',    22500.0)
        self.declare_parameter('container_marker_area', 60000.0)
        self.declare_parameter('turn_gain',             30.0)
        self.declare_parameter('forward_gain',          3.75)
        self.declare_parameter('forward_speed',         10.0)
        self.declare_parameter('cx_threshold',          0.15)
        self.declare_parameter('min_turn',              7.0)
        self.declare_parameter('grasp_distance_m',      0.15)
        self.declare_parameter('deploy_distance_m',     0.15)
        self.declare_parameter('use_distance_target',   True)
        self.declare_parameter('use_distance_trailer',  True)
        self.declare_parameter('approach_dist_gain', 10.0)   # s/m
        self.declare_parameter('approach_step_min',  1.0)    # s
        self.declare_parameter('approach_step_max',  3.0)    # s
        # Release the target lock if the locked ID has not been seen for
        # this long while still searching for it (0 = never release).
        self.declare_parameter('lock_timeout_s',     0.0)

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
        self.approach_dist_gain    = self.get_parameter('approach_dist_gain').value
        self.approach_step_min     = self.get_parameter('approach_step_min').value
        self.approach_step_max     = self.get_parameter('approach_step_max').value
        self.lock_timeout_s        = self.get_parameter('lock_timeout_s').value

        self.use_distance_target  = self.get_parameter('use_distance_target').value
        self.use_distance_trailer = self.get_parameter('use_distance_trailer').value

        if XGO_AVAILABLE:
            self.xgo = XGO(port='/dev/ttyAMA0')
            self.xgo.stop()
            self.xgo.translation("z", 80)
            self.xgo.attitude("p", 15)
        else:
            self.xgo = None
            self.get_logger().warn("XGO not available — motor commands suppressed")

        self.state               = State.SEARCH
        self.prev_state          = None
        self.motion_active       = False
        self.substep             = 0
        self.searching_container = False
        self.get_logger().info("[FSM] Starting new FSM (build: lock + deploy-verify)")

        # ── Marker list + lock ────────────────────────────────────────
        self.markers          = []      # latest list from camera_node
        self.markers_rx       = None    # receive time of that list
        self.locked_target_id = None    # int, locked in SEARCH
        self.locked_trailer   = None    # trailer group name, locked in SEARCH
        self.last_seen_locked = None    # time the locked marker was last seen
        self.tracked          = None    # current entry of the locked marker
        self._verify_t0       = None
        self.verify_phase     = "grasp"  # "grasp" | "deploy"

        # Values of the tracked marker used by ALIGN / APPROACH
        self.last_cx   = 0.0
        self.last_area = 0.0
        self.last_dist = -1.0

        # ── Subscriptions ─────────────────────────────────────────────
        self.create_subscription(String, 'aruco/markers', self._cb_markers, 10)
        # Dashboard (laptop) → FSM: switch mode.
        self.create_subscription(String, 'fsm/control', self._cb_control, 10)

        self.state_pub   = self.create_publisher(String, 'robot/state',       10)
        self.mode_pub    = self.create_publisher(String, 'fsm/control_mode',  10)
        self.battery_pub = self.create_publisher(Float32, 'robot/battery',    10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.create_timer(10.0, self._pub_battery)
        self._pub_battery()

    # ── Callbacks ────────────────────────────────────────────────────

    def _cb_markers(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"[FSM] markers parse error: {e}")
            return
        self.markers    = payload.get("markers", [])
        self.markers_rx = self.get_clock().now()
        self._update_tracked()

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
            if "lock_timeout_s" in cmd:
                self.lock_timeout_s = float(cmd["lock_timeout_s"])
            if cmd.get("release_lock"):
                self.get_logger().info("[FSM] Lock released via fsm/control")
                self._release_locks()
        except Exception as e:
            self.get_logger().error(f"[FSM] control parse error: {e}")

    # ── Lock helpers ──────────────────────────────────────────────────

    def _visible(self, kind, include_persisted=True):
        return [m for m in self.markers
                if m.get("kind") == kind and (include_persisted or not m.get("persisted"))]

    def _update_tracked(self):
        """Pick the entry of the locked marker from the current list."""
        if self.searching_container:
            cands = [m for m in self._visible("trailer")
                     if self.locked_trailer is not None and m.get("name") == self.locked_trailer]
            # Prefer the near marker over the far fallback, then the largest.
            cands.sort(key=lambda m: (bool(m.get("far")), -float(m.get("area", 0.0))))
        else:
            cands = [m for m in self._visible("target")
                     if self.locked_target_id is not None and int(m.get("id", -1)) == self.locked_target_id]

        self.tracked = cands[0] if cands else None
        if self.tracked is not None:
            self.last_seen_locked = self.get_clock().now()
            self.last_cx   = float(self.tracked.get("cx_f", self.tracked.get("cx", 0.0)))
            self.last_area = float(self.tracked.get("area_f", self.tracked.get("area", 0.0)))
            self.last_dist = float(self.tracked.get("dist", -1.0))
        else:
            self.last_dist = -1.0

    def _try_lock(self):
        """Lock the first marker of the active kind if nothing is locked yet."""
        if self.searching_container:
            if self.locked_trailer is None:
                trailers = self._visible("trailer", include_persisted=False)
                if trailers:
                    self.locked_trailer   = trailers[0]["name"]
                    self.last_seen_locked = self.get_clock().now()
                    self.get_logger().info(f"[FSM] Locked trailer '{self.locked_trailer}'")
        else:
            if self.locked_target_id is None:
                targets = self._visible("target", include_persisted=False)
                if targets:
                    self.locked_target_id = int(targets[0]["id"])
                    self.last_seen_locked = self.get_clock().now()
                    self.get_logger().info(
                        f"[FSM] Locked target id {self.locked_target_id} ('{targets[0]['name']}')")
        self._update_tracked()

    def _release_locks(self):
        self.locked_target_id = None
        self.locked_trailer   = None
        self.last_seen_locked = None
        self.tracked          = None
        self.last_dist        = -1.0

    def _marker_found(self):
        return self.tracked is not None

    def _check_lock_timeout(self):
        """Only while still looking for the target, never while carrying it."""
        if self.lock_timeout_s <= 0.0 or self.searching_container:
            return
        if self.locked_target_id is None or self.last_seen_locked is None:
            return
        if self.state in (State.GRASP, State.VERIFY, State.DEPLOY, State.DONE):
            return
        age = (self.get_clock().now() - self.last_seen_locked).nanoseconds * 1e-9
        if age > self.lock_timeout_s:
            self.get_logger().warn(
                f"[FSM] Target id {self.locked_target_id} not seen for {age:.1f}s → lock released")
            self._release_locks()

    # ── Motion helpers ────────────────────────────────────────────────

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

    # ── Trigger helpers ───────────────────────────────────────────────

    def _target_trigger(self):
        if self.use_distance_target:
            result = self.last_dist > 0.0 and self.last_dist <= self.grasp_distance_m
            self.get_logger().info(
                f"[FSM] _target_trigger [distance] id={self.locked_target_id} "
                f"dist={self.last_dist:.3f}m  threshold={self.grasp_distance_m}m  → {result}")
            return result
        else:
            result = self.last_area >= self.target_area
            self.get_logger().info(
                f"[FSM] _target_trigger [area] id={self.locked_target_id} "
                f"area={self.last_area:.0f}  threshold={self.target_area:.0f}  → {result}")
            return result

    def _trailer_trigger(self):
        if self.use_distance_trailer:
            result = self.last_dist > 0.0 and self.last_dist <= self.deploy_distance_m
            self.get_logger().info(
                f"[FSM] _trailer_trigger [distance] trailer={self.locked_trailer} "
                f"dist={self.last_dist:.3f}m  threshold={self.deploy_distance_m}m  → {result}")
            return result
        else:
            result = self.last_area >= self.container_target_area
            self.get_logger().info(
                f"[FSM] _trailer_trigger [area] trailer={self.locked_trailer} "
                f"area={self.last_area:.0f}  threshold={self.container_target_area:.0f}  → {result}")
            return result

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

    # ── Control loop ─────────────────────────────────────────────────
    def control_loop(self):
        s_msg = String(); s_msg.data = self.state.name
        self.state_pub.publish(s_msg)

        mode = {
            "use_distance_target":  self.use_distance_target,
            "use_distance_trailer": self.use_distance_trailer,
            "grasp_distance_m":     self.grasp_distance_m,
            "deploy_distance_m":    self.deploy_distance_m,
            "target_distance":      self.last_dist if not self.searching_container else -1.0,
            "trailer_distance":     self.last_dist if self.searching_container else -1.0,
            "approach_dist_gain":   self.approach_dist_gain,
            "approach_step_min":    self.approach_step_min,
            "approach_step_max":    self.approach_step_max,
            "lock_timeout_s":       self.lock_timeout_s,
            "locked_target_id":     self.locked_target_id,
            "locked_trailer":       self.locked_trailer,
            "searching_container":  self.searching_container,
        }
        m_msg = String(); m_msg.data = json.dumps(mode)
        self.mode_pub.publish(m_msg)

        if self.motion_active:
            if not self.motion_done():
                return

        if self.state != self.prev_state:
            self.get_logger().info(f"[FSM] {self.prev_state} → {self.state}")
            self.prev_state = self.state

        self._check_lock_timeout()

        # SEARCH
        if self.state == State.SEARCH:
            self._try_lock()
            if self._marker_found():
                self.state = State.ALIGN
                return
            self.xgo.turn(25)
            self.start_motion(2.0)
            self.state = State.SEARCH_TURNING

        elif self.state == State.SEARCH_TURNING:
            if self.motion_done():
                self.xgo.stop()
                self.start_motion(2.0)
                self.state = State.SEARCH_WAIT

        elif self.state == State.SEARCH_WAIT:
            if self.motion_done():
                self.state = State.SEARCH

        # ALIGN
        elif self.state == State.ALIGN:
            if not self._marker_found():
                self.state = State.SEARCH
                return

            cx_error = (self.last_cx - 960.0) / 960.0

            if abs(cx_error) > self.cx_threshold:
                if cx_error > 0:
                    turn   = min((-self.turn_gain * cx_error), -(self.min_turn))
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

        # APPROACH
        elif self.state == State.APPROACH:
            if not self._marker_found():
                self.xgo.move("x", -self.forward_speed)
                self.start_motion(1.0)
                self.state = State.APPROACH_BACK
                return

            if not self.searching_container:
                if self._target_trigger():
                    self.get_logger().info(
                        f"[FSM] Target trigger reached (id {self.locked_target_id}) → GRASP")
                    self.state = State.GRASP
                    return
            else:
                if self._trailer_trigger():
                    self.get_logger().info(
                        f"[FSM] Trailer trigger reached ('{self.locked_trailer}') → DEPLOY")
                    self.state = State.DEPLOY
                    return

            if not self.searching_container:
                use_dist = self.use_distance_target
                thresh   = self.grasp_distance_m
                area_tgt = self.target_area
            else:
                use_dist = self.use_distance_trailer
                thresh   = self.deploy_distance_m
                area_tgt = self.container_target_area
            dist_val = self.last_dist
            area_val = self.last_area

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

        # GRASP
        elif self.state == State.GRASP:
            if self.substep == 0 and self.motion_done():
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
                self.xgo.arm(83, -90)
                self.start_motion(1.0)
                self.substep = 4
            elif self.substep == 4 and self.motion_done():
                self.xgo.claw(255)
                self.start_motion(2.0)
                self.substep = 5
            elif self.substep == 5 and self.motion_done():
                self.xgo.reset()
                self.start_motion(1.0)
                self.substep = 6
            elif self.substep == 6 and self.motion_done():
                self.get_logger().info(
                    f"[FSM] Grasp of id {self.locked_target_id} finished → VERIFY (grasp)")
                self.substep      = 0
                self.verify_phase = "grasp"
                self.state        = State.VERIFY

        # VERIFY
        elif self.state == State.VERIFY:
            if self.substep == 0 and self.motion_done():
                self.xgo.translation("z", 80)
                self.xgo.attitude("p", 15)
                self.start_motion(1.0)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.move("x", -self.forward_speed)
                self.start_motion(1.0)
                self.substep = 2
            elif self.substep == 2 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                # Only marker lists received after the robot stopped count.
                self._verify_t0 = self.get_clock().now()
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                if self.markers_rx is None or self.markers_rx <= self._verify_t0:
                    return  # wait for a fresh marker list
                still_visible = any(
                    int(m.get("id", -1)) == self.locked_target_id
                    for m in self._visible("target", include_persisted=False)
                )
                verified = not still_visible
                self.get_logger().info(
                    f"[FSM] Verify ({self.verify_phase}) id {self.locked_target_id}: "
                    f"still visible={still_visible} → verified={verified}")
                self.substep = 0
                self.forward_speed = self.get_parameter('forward_speed').value

                if self.verify_phase == "deploy":
                    if verified:
                        # Object is in the trailer → finish as usual.
                        self.get_logger().info(
                            f"[FSM] Deploy verified, id {self.locked_target_id} gone → DONE")
                        self.state = State.DONE
                    else:
                        # Object lies next to the trailer → pick it up again.
                        self.get_logger().warn(
                            f"[FSM] Deploy failed, id {self.locked_target_id} still visible → ALIGN")
                        self.searching_container = False
                        self.locked_trailer      = None
                        self._update_tracked()
                        self.state = State.ALIGN
                    return

                if verified:
                    self.xgo.translation("z", 100)
                    self.xgo.attitude("p", 0)
                    self.start_motion(2.0)
                    # Keep the target lock (we carry it); trailer gets locked in SEARCH.
                    self.searching_container = True
                    self.locked_trailer      = None
                    self._update_tracked()
                    self.state               = State.SEARCH
                else:
                    # Retry the same locked ID.
                    self.xgo.translation("z", 80)
                    self.xgo.attitude("p", 15)
                    self.start_motion(2.0)
                    self._update_tracked()
                    self.state = State.SEARCH

        # DEPLOY
        elif self.state == State.DEPLOY:
            if self.substep == 0 and self.motion_done():
                self.xgo.move("x", 5)
                self.start_motion(1.5)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 2
            elif self.substep == 2 and self.motion_done():
                self.xgo.arm(130, -20)
                self.start_motion(1.0)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                self.xgo.claw(0)
                self.start_motion(1.0)
                self.substep = 4
            elif self.substep == 4 and self.motion_done():
                self.xgo.reset()
                self.start_motion(1.0)
                self.substep = 5
            elif self.substep == 5 and self.motion_done():
                self.xgo.translation("z", 0)
                self.xgo.attitude("p", 0)
                self.start_motion(2.0)
                self.substep = 6
            elif self.substep == 6 and self.motion_done():
                self.get_logger().info(
                    f"[FSM] Deploy of id {self.locked_target_id} finished → VERIFY (deploy)")
                self.substep      = 0
                self.verify_phase = "deploy"
                self.state        = State.VERIFY

        # DONE
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
                self.get_logger().info(
                    f"[FSM] Deploy of id {self.locked_target_id} complete → releasing locks")
                self._release_locks()
                self.state               = State.SEARCH
                self.prev_state          = None
                self.motion_active       = False
                self.searching_container = False
                self.substep             = 0
                self.get_logger().info("[FSM] Restarted FSM")


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
