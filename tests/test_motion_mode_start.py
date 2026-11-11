"""Motion-mode (standing) start/exit safety, exercised with in-memory DDS and no threads."""

import ast
import copy
import threading
import time
import unittest
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Measured on the standing robot (2026-11-11): legs 0..11.
STANDING_LEGS = [-0.38, -0.04, 0.03, 0.61, -0.27, 0.07, -0.37, 0.03, -0.02, 0.61, -0.31, 0.03]
# Lowest seated knee seen in the button-press datasets is 1.41 rad.
SEATED_LEGS = [-1.43, 0.0, 0.0, 1.41, -0.3, 0.0, -1.42, 0.0, 0.0, 1.41, -0.3, 0.0]


class PublishedFrame(Exception):
    pass


class MemoryPublisher:
    def __init__(self, topic, message_type):
        self.topic = topic
        self.messages = []
        self.stop_after_write = False

    def Init(self):
        pass

    def Write(self, message):
        self.messages.append(copy.deepcopy(message))
        if self.stop_after_write:
            raise PublishedFrame()


class MemorySubscriber:
    def __init__(self, *args):
        pass

    def Init(self):
        pass

    def Read(self, *args):
        return SimpleNamespace(mode_machine=0)


class NoThread:
    """Feed one lowstate (q_i = 0.01 * i) instead of starting the subscribe/publish threads."""

    def __init__(self, target):
        self.target = target
        self.daemon = True

    def start(self):
        if self.target.__name__ == "_subscribe_motor_state":
            self.target.__self__.lowstate_buffer.SetData(SimpleNamespace(
                mode_machine=0,
                motor_state=[SimpleNamespace(q=0.01 * i, dq=0.0) for i in range(35)],
            ))


def motor_command():
    return SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)


class RemoveImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def load_module(path, namespace):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.If)]  # skip __main__ demos
    exec(compile(RemoveImports().visit(tree), str(path), "exec"), namespace)
    return namespace


def arm_code():
    return load_module("teleop/robot_control/robot_arm.py", {
        "np": np, "IntEnum": IntEnum,
        "time": SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda s: None),
        "threading": SimpleNamespace(Thread=NoThread, Lock=threading.Lock),
        "logging_mp": SimpleNamespace(getLogger=lambda name: Mock()),
        "ChannelPublisher": MemoryPublisher, "ChannelSubscriber": MemorySubscriber,
        "ChannelFactoryInitialize": Mock(),
        "hg_LowCmd": object, "hg_LowState": object, "go_LowCmd": object, "go_LowState": object,
        "CRC": lambda: SimpleNamespace(Crc=lambda message: 0),
        "unitree_hg_msg_dds__LowCmd_": lambda: SimpleNamespace(
            mode_pr=0, mode_machine=0, crc=0, motor_cmd=[motor_command() for _ in range(35)]),
        "unitree_go_msg_dds__LowCmd_": lambda: SimpleNamespace(motor_cmd=[motor_command() for _ in range(35)]),
    })


MEASURED_ARMS = [0.01 * i for i in range(15, 29)]


class MotionStartTests(unittest.TestCase):
    def setUp(self):
        self.code = arm_code()

    def first_packet(self, controller):
        controller.lowcmd_publisher.stop_after_write = True
        with self.assertRaises(PublishedFrame):
            controller._ctrl_motor_state()
        return controller.lowcmd_publisher.messages[0]

    def test_motion_without_initial_pose_holds_measured_arms(self):
        controller = self.code["G1_29_ArmController"](motion_mode=True, initial_target_q=None)
        self.assertEqual(controller.lowcmd_publisher.topic, "rt/arm_sdk")
        np.testing.assert_allclose(controller.initial_target_q, MEASURED_ARMS)
        packet = self.first_packet(controller)
        np.testing.assert_allclose([packet.motor_cmd[i].q for i in range(15, 29)], MEASURED_ARMS)
        self.assertEqual(packet.motor_cmd[29].q, 1.0)
        for i in range(15):  # waist/legs held at the measured startup angles
            self.assertAlmostEqual(packet.motor_cmd[i].q, 0.01 * i)

    def test_explicit_initial_pose_is_still_used(self):
        target = np.linspace(0.1, 0.5, 14)
        controller = self.code["G1_29_ArmController"](motion_mode=False, initial_target_q=target)
        self.assertFalse(controller.hold_startup_arms)
        np.testing.assert_allclose(controller.initial_target_q, target)
        np.testing.assert_allclose(controller.q_target, target)

    def test_motion_go_home_is_slow_and_releases_weight(self):
        controller = self.code["G1_29_ArmController"](motion_mode=True, initial_target_q=None)
        controller.msg.motor_cmd[29].q = 1.0
        controller._speed_gradual_max = True
        controller.ctrl_dual_arm_go_home()
        self.assertFalse(controller._speed_gradual_max)
        self.assertEqual(controller.arm_velocity_limit, self.code["MOTION_GO_HOME_VELOCITY_LIMIT"])
        np.testing.assert_allclose(controller.q_target, MEASURED_ARMS)
        self.assertEqual(controller.msg.motor_cmd[29].q, 0.0)

    def test_motion_go_home_releases_weight_even_when_blocked(self):
        controller = self.code["G1_29_ArmController"](motion_mode=True, initial_target_q=None)
        controller.initial_target_q = controller.initial_target_q + 0.3  # never reachable
        controller.msg.motor_cmd[29].q = 1.0
        controller.ctrl_dual_arm_go_home()
        self.assertEqual(controller.msg.motor_cmd[29].q, 0.0)

    def test_debug_go_home_never_touches_weight(self):
        controller = self.code["G1_29_ArmController"](motion_mode=False, initial_target_q=np.ones(14))
        before = controller.msg.motor_cmd[29].q
        controller.ctrl_dual_arm_go_home()
        self.assertEqual(controller.msg.motor_cmd[29].q, before)
        self.assertEqual(controller.arm_velocity_limit, 20.0)


class StandingGuardTests(unittest.TestCase):
    def setUp(self):
        self.code = load_module("teleop/utils/motion_switcher.py", {"time": time})

    def test_standing_legs_refuse_debug_mode(self):
        self.assertTrue(self.code["legs_look_standing"](STANDING_LEGS))
        message = self.code["debug_mode_refusal"](STANDING_LEGS)
        self.assertIn("--motion", message)

    def test_seated_legs_allow_debug_mode(self):
        self.assertFalse(self.code["legs_look_standing"](SEATED_LEGS))
        self.assertIsNone(self.code["debug_mode_refusal"](SEATED_LEGS))


class MainScriptWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "teleop/teleop_hand_and_arm.py").read_text()
        ast.parse(self.source)

    def test_motion_mode_skips_initial_pose(self):
        self.assertIn("initial_target_q = None if args.motion else resolve_initial_target_q(args)", self.source)

    def test_standing_check_runs_before_releasing_motion_service(self):
        guard = self.source.index("debug_mode_refusal(read_leg_q())")
        release = self.source.index("motion_switcher.Enter_Debug_Mode()")
        self.assertLess(guard, release)

    def test_motion_mode_requires_running_motion_service(self):
        branch = self.source[self.source.index("        if args.motion:\n            logger_mp.info(\"Motion mode"):]
        self.assertLess(branch.index("CheckMode()"), branch.index("else:"))


def build_parser():
    """Rebuild only the script's argparse parser (no robot, camera or XR imports)."""
    import argparse
    lines = (ROOT / "teleop/teleop_hand_and_arm.py").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "parser = argparse.ArgumentParser(" in l)
    end = next(i for i, l in enumerate(lines) if "args = parser.parse_args()" in l)
    namespace = {"argparse": argparse, "INITIAL_TARGET_Q_CONFIG_PATH": "unused"}
    exec("\n".join(l[4:] for l in lines[start:end]), namespace)
    return namespace["parser"]


class NoCameraTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "teleop/teleop_hand_and_arm.py").read_text()

    def test_flag_parses_with_motion_keyboard_command(self):
        args = build_parser().parse_args("--motion --no-camera --keyboard-arm left --headless".split())
        self.assertTrue(args.no_camera)
        self.assertFalse(args.record)

    def test_config_disables_every_stream_and_has_keys_the_script_reads(self):
        tree = ast.parse(self.source)
        node = next(n for n in tree.body if getattr(n, "name", None) == "no_camera_config")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "no_camera_config", "exec"), namespace)
        config = namespace["no_camera_config"]()
        head = config["head_camera"]
        for key in ("enable_zmq", "enable_webrtc", "binocular", "image_shape", "webrtc_port"):
            self.assertIn(key, head)
        for name in ("head_camera", "left_wrist_camera", "right_wrist_camera", "right_side_camera"):
            self.assertFalse(config[name]["enable_zmq"])
            self.assertFalse(config[name]["enable_webrtc"])

    def test_image_server_is_only_contacted_with_cameras(self):
        branch = self.source[self.source.index("        if args.no_camera:\n            camera_config = no_camera_config()"):]
        self.assertLess(branch.index("else:"), branch.index("ImageClient("))
        self.assertIn("display_mode='pass-through' if args.no_camera else args.display_mode", self.source)

    def test_recording_without_cameras_is_rejected(self):
        check = self.source.index("if args.no_camera and args.record:")
        self.assertLess(self.source.index("args = parser.parse_args()"), check)
        self.assertLess(check, self.source.index("ChannelFactoryInitialize(0"))


def load_main_definitions(*names):
    """Load selected functions/classes/constants from the main script without its imports."""
    tree = ast.parse((ROOT / "teleop/teleop_hand_and_arm.py").read_text())
    wanted = []
    for node in tree.body:
        if getattr(node, "name", None) in names:
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            wanted.append(node)
    namespace = {"np": np}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "teleop_hand_and_arm.py", "exec"), namespace)
    return namespace


KEYBOARD_NAMES = (
    "rotation_matrix_from_rotvec", "rotvec_from_rotation_matrix", "orthonormalize_rotation",
    "approach_velocity", "KeyboardEndEffectorController",
    "KEYBOARD_TRANSLATION_KEYS", "KEYBOARD_ROTATION_KEYS", "KEYBOARD_RETURN_HOME_KEYS",
)


class KeyboardSmoothingTests(unittest.TestCase):
    def setUp(self):
        self.code = load_main_definitions(*KEYBOARD_NAMES)

    def controller(self, **kwargs):
        return self.code["KeyboardEndEffectorController"](
            control_arm="right", initial_left_target=np.eye(4), initial_right_target=np.eye(4), **kwargs)

    def test_zero_accel_keeps_instant_start_and_stop(self):
        ctrl = self.controller(linear_speed=0.025, angular_speed=0.3)
        _, right, _ = ctrl.update(None, None, 0.1, {"w"})
        self.assertAlmostEqual(np.linalg.norm(right[:3, 3]), 0.0025)
        _, right_after, _ = ctrl.update(None, None, 0.1, set())
        np.testing.assert_allclose(right_after, right)

    def test_accel_ramps_up_and_coasts_to_a_stop(self):
        ctrl = self.controller(linear_speed=0.05, angular_speed=0.5, linear_accel=0.15, angular_accel=1.5)
        dt = 1 / 30
        speeds = []
        for _ in range(15):
            ctrl.update(None, None, dt, {"w"})
            speeds.append(np.linalg.norm(ctrl.linear_velocity))
        self.assertAlmostEqual(speeds[0], 0.15 * dt)
        self.assertTrue(all(b >= a for a, b in zip(speeds, speeds[1:])))
        self.assertAlmostEqual(speeds[-1], 0.05)
        stop_steps = 0
        while np.linalg.norm(ctrl.linear_velocity) > 0 and stop_steps < 100:
            ctrl.update(None, None, dt, set())
            stop_steps += 1
        self.assertIn(stop_steps, (10, 11))  # 0.05 / 0.15 = 0.33 s, plus float rounding

    def test_short_key_dropout_no_longer_stops_the_arm(self):
        ctrl = self.controller(linear_speed=0.05, angular_speed=0.5, linear_accel=0.15, angular_accel=1.5)
        for _ in range(15):
            ctrl.update(None, None, 1 / 30, {"w"})
        ctrl.update(None, None, 1 / 30, set())  # one spurious release frame
        self.assertGreater(np.linalg.norm(ctrl.linear_velocity), 0.04)

    def test_return_home_clears_velocity(self):
        ctrl = self.controller(linear_speed=0.05, angular_speed=0.5, linear_accel=0.15, angular_accel=1.5)
        for _ in range(5):
            ctrl.update(None, None, 1 / 30, {"w", "j"})
        ctrl.update(None, None, 1 / 30, set(), return_home_active=True)
        self.assertFalse(np.any(ctrl.linear_velocity))
        self.assertFalse(np.any(ctrl.angular_velocity))


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def time(self):
        return self.now


class JointInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.code = arm_code()
        self.code["time"] = SimpleNamespace(time=self.clock.time, monotonic=self.clock.time, sleep=lambda s: None)

    def published_arm_q(self, controller):
        controller.lowcmd_publisher.stop_after_write = True
        with self.assertRaises(PublishedFrame):
            controller._ctrl_motor_state()
        packet = controller.lowcmd_publisher.messages[-1]
        return np.array([packet.motor_cmd[i].q for i in range(15, 29)])

    def test_targets_are_interpolated_between_frames(self):
        controller = self.code["G1_29_ArmController"](motion_mode=True, target_interp_time=0.1)
        start = controller.get_current_dual_arm_q()
        target = start + 0.01
        controller.ctrl_dual_arm(target, np.zeros(14))
        self.clock.now += 0.05
        np.testing.assert_allclose(self.published_arm_q(controller), start + 0.005)
        self.clock.now += 0.2
        np.testing.assert_allclose(self.published_arm_q(controller), target)

    def test_new_target_continues_from_the_current_interpolated_point(self):
        controller = self.code["G1_29_ArmController"](motion_mode=True, target_interp_time=0.1)
        start = controller.get_current_dual_arm_q()
        controller.ctrl_dual_arm(start + 0.01, np.zeros(14))
        self.clock.now += 0.05
        controller.ctrl_dual_arm(start + 0.02, np.zeros(14))
        np.testing.assert_allclose(controller._interp_from, start + 0.005)

    def test_debug_full_body_arms_are_interpolated_too(self):
        controller = self.code["G1_29_ArmController"](
            motion_mode=False, initial_target_q=np.array(MEASURED_ARMS),
            initial_full_body_q=np.array([0.01 * i for i in range(35)]),
            control_full_body=True, body_ramp_duration=0.0, target_interp_time=0.1)
        start = controller.get_current_dual_arm_q()
        controller.ctrl_dual_arm(start + 0.01, np.zeros(14))
        self.clock.now += 0.05
        np.testing.assert_allclose(self.published_arm_q(controller), start + 0.005)
        self.clock.now += 0.2
        np.testing.assert_allclose(self.published_arm_q(controller), start + 0.01)

    def test_disabled_interpolation_jumps_to_target(self):
        controller = self.code["G1_29_ArmController"](motion_mode=True)
        target = controller.get_current_dual_arm_q() + 0.01
        controller.ctrl_dual_arm(target, np.zeros(14))
        np.testing.assert_allclose(self.published_arm_q(controller), target)


class TuningDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.code = load_main_definitions("KEYBOARD_TUNING_DEFAULTS", "resolve_keyboard_tuning")
        self.parser = build_parser()

    def resolved(self, argv):
        return self.code["resolve_keyboard_tuning"](self.parser.parse_args(argv.split()))

    def test_debug_keeps_speed_and_joint_stepping_but_smooths_keys(self):
        args = self.resolved("--headless")
        # Speed and 30 Hz joint stepping match the seated dataset and inference; key start/stop is smoothed.
        self.assertEqual((args.keyboard_linear_speed, args.keyboard_angular_speed), (0.025, 0.30))
        self.assertGreater(args.keyboard_linear_accel, 0)
        self.assertGreater(args.keyboard_release_gap, 0.09)
        self.assertEqual(args.keyboard_repeat_delay, 0.75)
        self.assertFalse(args.smooth_joint_targets)

    def test_debug_interpolation_can_be_enabled(self):
        self.assertTrue(self.resolved("--headless --smooth-joint-targets").smooth_joint_targets)

    def test_motion_defaults_are_faster_and_smoothed(self):
        args = self.resolved("--motion --headless")
        self.assertEqual((args.keyboard_linear_speed, args.keyboard_angular_speed), (0.05, 0.50))
        self.assertGreater(args.keyboard_linear_accel, 0)
        self.assertGreater(args.keyboard_release_gap, 0.09)  # above a 90 ms OS repeat interval
        self.assertTrue(args.smooth_joint_targets)

    def test_explicit_values_win(self):
        args = self.resolved("--motion --keyboard-linear-speed 0.08 --keyboard-linear-accel 0 --no-smooth-joint-targets")
        self.assertEqual(args.keyboard_linear_speed, 0.08)
        self.assertEqual(args.keyboard_linear_accel, 0.0)
        self.assertFalse(args.smooth_joint_targets)


if __name__ == "__main__":
    unittest.main()
