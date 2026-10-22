"""Feedforward selection and dispatch without DDS or physical robot motion."""

import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("episode_replay", ROOT / "scripts/replay_episode.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def frame(q, torque=None):
    values = {
        "left_arm": {"qpos": list(q[:7]), "torque": []},
        "right_arm": {"qpos": list(q[7:]), "torque": []},
    }
    actions = copy.deepcopy(values)
    if torque is not None:
        actions["left_arm"]["torque"] = list(torque[:7])
        actions["right_arm"]["torque"] = list(torque[7:])
    return {"states": values, "actions": actions}


class EpisodeFeedforwardTests(unittest.TestCase):
    def setUp(self):
        self.q = np.linspace(0.1, 0.2, 14)
        self.gravity = Mock(side_effect=lambda q: np.asarray(q) * 2 + 1)
        self.model = patch.object(replay, "create_gravity_solver", return_value=self.gravity)
        self.model.start()
        self.addCleanup(self.model.stop)

    def test_old_data_uses_gravity_and_new_action_data_uses_recorded_torque(self):
        frames = [frame(self.q), frame(self.q, np.arange(14) + 3)]
        _, torques = replay.prepare_arm_torques(frames, [self.q] * 2, "actions", "auto", "G1_29")
        np.testing.assert_allclose(torques[0], self.q * 2 + 1)
        np.testing.assert_array_equal(torques[1], np.arange(14) + 3)
        # An explicitly recorded zero is valid, distinct from missing torque.
        _, torques = replay.prepare_arm_torques(
            [frame(self.q, np.zeros(14))], [self.q], "actions", "auto", "G1_29"
        )
        np.testing.assert_array_equal(torques[0], np.zeros(14))

    def test_state_replay_never_uses_measured_or_action_torques(self):
        data = frame(self.q, np.full(14, 999))
        data["states"]["left_arm"]["torque"] = [888] * 7
        data["states"]["right_arm"]["torque"] = [888] * 7
        _, torques = replay.prepare_arm_torques([data], [self.q], "states", "auto", "G1_29")
        np.testing.assert_allclose(torques[0], self.q * 2 + 1)

    def test_explicit_gravity_and_zero_override_recorded_values(self):
        data = frame(self.q, np.full(14, 999))
        for mode in ("gravity", "zero"):
            with self.subTest(mode=mode):
                solve, torques = replay.prepare_arm_torques([data], [self.q], "actions", mode, "G1_29")
                expected = self.q * 2 + 1 if mode == "gravity" else np.zeros(14)
                np.testing.assert_allclose(torques[0], expected)
                np.testing.assert_allclose(solve(self.q), expected)

    def execute(self, frames, dds=None):
        controller = Mock()
        controller.get_current_dual_arm_q.return_value = np.zeros(14)
        dds = dds if dds is not None else Mock()
        with (
            patch.object(replay, "initialize_dds", dds),
            patch.object(replay, "create_arm_controller", return_value=controller),
            patch.object(replay, "sleep_to_rate"),
        ):
            replay.execute_robot_replay(
                episode_dir=Path("/unused"), frames=frames, camera_keys=[],
                initial_target_q=np.full(14, 0.05), fps=2, speed=1,
                source="actions", arm_name="G1_29", ee_name="none",
                motion=True, sim=False, network_interface=None, ramp_time=1,
                max_initial_arm_delta=1.5, max_frame_arm_delta=0.35,
                allow_large_initial_jump=False, go_home=False,
                tile_width=640, show_gui=False,
            )
        dds.assert_called_once()
        return controller

    def test_hold_both_startup_ramps_and_replay_send_nonzero_feedforward(self):
        recorded = np.arange(14) + 3
        controller = self.execute([frame(self.q), frame(self.q, recorded)])
        calls = controller.ctrl_dual_arm.call_args_list
        self.assertEqual(len(calls), 7)  # hold, two ramps of two steps, two frames
        for call in calls[:-1]:
            q, tau = call.args
            np.testing.assert_allclose(tau, q * 2 + 1)
        np.testing.assert_array_equal(calls[-1].args[1], recorded)

    def test_bad_recorded_torque_aborts_before_dds(self):
        for bad in ([1], [float("nan")] * 7, [float("inf")] * 7):
            with self.subTest(bad=bad):
                dds = Mock()
                data = frame(self.q, np.ones(14))
                data["actions"]["left_arm"]["torque"] = bad
                with self.assertRaises(ValueError):
                    self.execute([data], dds=dds)
                dds.assert_not_called()

    def test_missing_recorded_torque_is_an_error_in_recorded_mode(self):
        with self.assertRaisesRegex(ValueError, "Missing recorded"):
            replay.prepare_arm_torques([frame(self.q)], [self.q], "actions", "recorded", "G1_29")

    def test_recorder_persists_the_sent_feedforward_in_action_fields(self):
        # Evaluate the real action payload without starting the teleoperation loop.
        tree = ast.parse((ROOT / "teleop/teleop_hand_and_arm.py").read_text())
        action_payload = next(node.value for node in ast.walk(tree)
                              if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "actions" for t in node.targets))
        sent = np.linspace(-3, 4, 14)
        payload = eval(compile(ast.Expression(action_payload), "recorded_actions", "eval"), {
            "left_arm_action": self.q[:7], "right_arm_action": self.q[7:],
            "left_hand_action": [], "right_hand_action": [],
            "current_body_action": [], "sol_tauff": sent,
        })
        np.testing.assert_array_equal(replay.extract_action_arm_tau({"actions": payload}, 14), sent)


if __name__ == "__main__":
    unittest.main()
