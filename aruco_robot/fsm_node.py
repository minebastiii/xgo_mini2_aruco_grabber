#!/usr/bin/env python3

import json
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Float32MultiArray, Float32, String
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

        self.use_distance_target  = self.get_parameter('use_distance_target').value
        self.use_distance_trailer = self.get_parameter('use_distance_trailer').value

        self.once = True

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
        self.picked_up_id        = -1
        self.get_logger().info("[FSM] Starting new FSM")

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

        # ── Subscriptions — now fed by camera_node.py on the robot ─────
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
        # Dashboard (laptop) → FSM: switch mode.
        self.create_subscription(String, 'fsm/control',
                                 self._cb_control, 10)

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
        except Exception as e:
            self.get_logger().error(f"[FSM] control parse error: {e}")

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
            "target_distance":      self._target_distance,
            "trailer_distance":     self._trailer_distance,
            "approach_dist_gain":   self.approach_dist_gain,
            "approach_step_min":    self.approach_step_min,
            "approach_step_max":    self.approach_step_max,
        }
        m_msg = String(); m_msg.data = json.dumps(mode)
        self.mode_pub.publish(m_msg)

        if self.motion_active:
            if not self.motion_done():
                return

        if self.state != self.prev_state:
            self.get_logger().info(f"[FSM] {self.prev_state} → {self.state}")
            self.prev_state = self.state

        # SEARCH
        if self.state == State.SEARCH:
            if self.last_found:
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
            if not self.last_found:
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
            if not self.last_found:
                self.xgo.move("x", -self.forward_speed)
                self.start_motion(1.0)
                self.state = State.APPROACH_BACK
                return

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
                self.substep      = 0
                self.picked_up_id = self.last_id
                self.last_id      = -1
                self.state        = State.VERIFY

        # VERIFY
        elif self.state == State.VERIFY:
            if self.substep == 0:
                self.xgo.translation("z", 80)
                self.xgo.attitude("p", 15)
                self.start_motion(1.0)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.move("x", -self.forward_speed)
                self.start_motion(1.0)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 4
            elif self.substep == 4 and self.motion_done():
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

        # DEPLOY
        elif self.state == State.DEPLOY:
            if self.substep == 0 and self.motion_done():
                self.xgo.move("y", 5)
                self.start_motion(1.5)
                self.substep = 1
            elif self.substep == 1 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 2
            elif self.substep == 2 and self.motion_done():
                self.xgo.move("x", 5)
                self.start_motion(1.5)
                self.substep = 3
            elif self.substep == 3 and self.motion_done():
                self.xgo.stop()
                self.start_motion(1.0)
                self.substep = 4
            elif self.substep == 4 and self.motion_done():
                self.xgo.arm(130, -20)
                self.start_motion(1.0)
                self.substep = 5
            elif self.substep == 5 and self.motion_done():
                self.xgo.claw(0)
                self.start_motion(1.0)
                self.substep = 6
            elif self.substep == 6 and self.motion_done():
                self.xgo.reset()
                self.start_motion(1.0)
                self.substep = 7
            elif self.substep == 7 and self.motion_done():
                self.xgo.translation("z", 0)
                self.xgo.attitude("p", 0)
                self.start_motion(2.0)
                self.substep = 9
            elif self.substep == 9 and self.motion_done():
                self.substep = 0
                self.state   = State.DONE

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
                self.state               = State.SEARCH
                self.prev_state          = None
                self.motion_active       = False
                self.searching_container = False
                self.picked_up_id        = -1
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
