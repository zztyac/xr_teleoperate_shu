"""Exercise actual replay/controller code with in-memory DDS and no threads."""

import ast
import copy
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from enum import IntEnum
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EPISODE = Path("/mnt/data/zty/json_data/trunk_open/episode_0074")


class PublishedFrame(Exception):
    pass


class MemoryPublisher:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.messages = []
        self.stop_after_write = False
        self.instances.append(self)

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

    def Read(self):
        return SimpleNamespace(mode_machine=0)


class NoThread:
    def __init__(self, target):
        self.target = target

    def start(self):
        if self.target.__name__ == "_subscribe_motor_state":
            self.target.__self__.lowstate_buffer.SetData(SimpleNamespace(
                motor_state=[SimpleNamespace(q=0.01 * i, dq=0.0) for i in range(35)],
            ))


def motor_command():
    return SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)


class RemoveImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def isolated_controllers():
    namespace = {
        "np": np, "IntEnum": IntEnum, "time": time,
        "threading": SimpleNamespace(Thread=NoThread, Lock=threading.Lock),
        "logging_mp": SimpleNamespace(getLogger=lambda name: Mock()),
        "ChannelPublisher": MemoryPublisher, "ChannelSubscriber": MemorySubscriber,
        "hg_LowCmd": object, "hg_LowState": object, "MotorCmds_": SimpleNamespace,
        "CRC": lambda: SimpleNamespace(Crc=lambda message: 0),
        "unitree_hg_msg_dds__LowCmd_": lambda: SimpleNamespace(
            motor_cmd=[motor_command() for _ in range(35)],
        ),
        "unitree_go_msg_dds__MotorCmd_": motor_command,
    }
    tree = ast.parse((ROOT / "teleop/robot_control/robot_arm.py").read_text())
    # Keep real definitions and constants; never execute the manual demo.
    tree.body = [node for node in tree.body if not isinstance(node, ast.If)]
    exec(compile(RemoveImports().visit(tree), "robot_arm.py", "exec"), namespace)
    tree = ast.parse((ROOT / "scripts/replay_robot_raw_state.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "BraincoDirectController"]
    exec(compile(RemoveImports().visit(tree), "replay_robot_raw_state.py", "exec"), namespace)
    return namespace


class MotionStateReplayTests(unittest.TestCase):
    def setUp(self):
        MemoryPublisher.instances = []
        self.code = isolated_controllers()

    def test_motion_packet_holds_startup_waist_while_arms_follow_targets(self):
        controller = self.code["G1_29_ArmController"](motion_mode=True)
        publisher = controller.lowcmd_publisher
        self.assertEqual(publisher.topic, "rt/arm_sdk")
        publisher.stop_after_write = True
        with self.assertRaises(PublishedFrame):
            controller._ctrl_motor_state()
        packet = publisher.messages[0]
        enabled = [i for i, command in enumerate(packet.motor_cmd) if command.mode]
        self.assertEqual(enabled, list(range(35)))
        for i in (12, 13, 14):
            self.assertEqual(packet.motor_cmd[i].q, 0.01 * i)
            self.assertEqual(packet.motor_cmd[i].kp, 300.0)
            self.assertEqual(packet.motor_cmd[i].kd, 3.0)
        self.assertEqual(packet.motor_cmd[29].q, 1.0)
        self.assertIsNone(controller.initial_body_q)
        target = controller.get_current_dual_arm_q() + 0.02
        controller.ctrl_dual_arm(target, np.zeros(14))
        with self.assertRaises(PublishedFrame):
            controller._ctrl_motor_state()
        following_packet = publisher.messages[-1]
        for i in (12, 13, 14):
            self.assertEqual(vars(following_packet.motor_cmd[i]), vars(packet.motor_cmd[i]))
        np.testing.assert_allclose([following_packet.motor_cmd[i].q for i in range(15, 29)], target)

    def test_debug_packet_behavior_is_preserved(self):
        controller = self.code["G1_29_ArmController"](motion_mode=False)
        self.assertEqual(controller.lowcmd_publisher.topic, "rt/lowcmd")
        self.assertTrue(all(command.mode == 1 for command in controller.msg.motor_cmd))

    @unittest.skipUnless((EPISODE / "data.json").is_file(), "Local episode unavailable")
    def test_all_episode_targets_reach_only_arm_and_hand_controllers(self):
        spec = importlib.util.spec_from_file_location("state_replay", ROOT / "scripts/replay_robot_raw_state.py")
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        with patch("sys.argv", ["replay", "--episode-path", str(EPISODE), "--motion", "--yes"]):
            args = replay.parse_args()
        payload, frames = replay.load_episode(EPISODE)
        for frame in frames:
            # Poison unused inputs: replay must never read body or actions.
            frame["states"]["body"] = object()
            frame["actions"] = object()
        controller = self.code["G1_29_ArmController"](motion_mode=True)
        controller.ctrl_dual_arm = Mock(wraps=controller.ctrl_dual_arm)
        hands = self.code["BraincoDirectController"]()
        with (
            patch.object(replay, "load_episode", return_value=(payload, frames)),
            patch.object(replay, "initialize_dds") as dds,
            patch.object(replay, "enter_debug_mode", side_effect=AssertionError("Debug mode entered")),
            patch.object(replay, "create_arm_controller", return_value=controller) as factory,
            patch.object(replay, "BraincoDirectController", return_value=hands),
            patch.object(replay.time, "sleep"),
        ):
            replay.replay(args)
        dds.assert_called_once_with(sim=False, network_interface=None)
        factory.assert_called_once_with("G1_29", motion=True, sim=False)
        self.assertEqual(controller.ctrl_dual_arm.call_count, len(frames) + 1)
        self.assertEqual(len(hands.left_pub.messages), len(frames))
        self.assertEqual(len(hands.right_pub.messages), len(frames))
        for frame, call, left, right in zip(
            frames, controller.ctrl_dual_arm.call_args_list[1:],
            hands.left_pub.messages, hands.right_pub.messages,
        ):
            np.testing.assert_array_equal(call.args[0], replay.arm_state_qpos(frame))
            np.testing.assert_array_equal(call.args[1], np.zeros(14))
            np.testing.assert_array_equal([cmd.q for cmd in left.cmds], frame["states"]["left_ee"]["qpos"])
            np.testing.assert_array_equal([cmd.q for cmd in right.cmds], frame["states"]["right_ee"]["qpos"])
        self.assertEqual(
            [publisher.topic for publisher in MemoryPublisher.instances],
            ["rt/arm_sdk", "rt/brainco/left/cmd", "rt/brainco/right/cmd"],
        )


if __name__ == "__main__":
    unittest.main()
