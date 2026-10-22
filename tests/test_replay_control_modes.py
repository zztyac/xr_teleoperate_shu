"""Check replay CLI interlocks without importing SDKs or connecting to hardware."""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "scripts/replay_robot_raw_state.py",
    "scripts/replay_robot_raw.py",
    "scripts/replay_episode.py",
    "teleop/replay_episode_real.py",
)


def parse_only(script, argv):
    # Extract the actual parser so even the legacy script's top-level robot
    # imports cannot initialize hardware during these regression checks.
    tree = ast.parse((ROOT / script).read_text())
    parser_function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "DEFAULT_EPISODE_PATH": Path("/unused"),
        "DEFAULT_EPISODE_DIR": Path("/unused"),
        "DEFAULT_TILE_WIDTH": 640,
        "DEFAULT_INITIAL_TARGET_Q_FILE": Path("/unused.json"),
        "ARM_CONTROLLER_BY_NAME": dict.fromkeys(["G1_29", "G1_23", "H1_2", "H1", "H2"]),
    }
    exec(compile(ast.Module(body=[parser_function], type_ignores=[]), script, "exec"), namespace)
    with patch.object(sys, "argv", [script, *argv]):
        return namespace["parse_args"]()


class ReplayControlModeTests(unittest.TestCase):
    def hardware_args(self, script):
        return ["--execute", "--yes"] if script.endswith("/replay_episode.py") else ["--yes"]

    def assert_blocked(self, script, argv, expected):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_only(script, argv)
        self.assertEqual(error.exception.code, 2)
        self.assertIn(expected, stderr.getvalue())

    def test_omitted_mode_is_blocked_even_with_yes(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                self.assert_blocked(script, self.hardware_args(script), "releasing locomotion")

    def test_skip_debug_mode_cannot_bypass_topic_interlock(self):
        for script in SCRIPTS[:2]:
            with self.subTest(script=script):
                self.assert_blocked(script, ["--yes", "--skip-debug-mode"], "full-body lowcmd")

    def test_motion_and_explicit_debug_are_separate_choices(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                motion = parse_only(script, self.hardware_args(script) + ["--motion"])
                self.assertTrue(motion.motion)
                self.assertFalse(motion.allow_debug_mode)
                debug = parse_only(script, self.hardware_args(script) + ["--allow-debug-mode"])
                self.assertFalse(debug.motion)
                self.assertTrue(debug.allow_debug_mode)
                self.assert_blocked(
                    script, ["--motion", "--allow-debug-mode"], "not allowed with argument",
                )

    def test_offline_and_simulator_use_need_no_hardware_mode(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                offline = [] if script.endswith("/replay_episode.py") else ["--dry-run"]
                self.assertFalse(parse_only(script, offline).motion)
                self.assertTrue(parse_only(script, self.hardware_args(script) + ["--sim"]).sim)

    def test_h1_cannot_claim_motion_mode_while_using_lowcmd(self):
        for script in SCRIPTS[2:]:
            with self.subTest(script=script):
                self.assert_blocked(
                    script, self.hardware_args(script) + ["--motion", "--arm", "H1"],
                    "H1 has no motion-mode arm controller",
                )

    def test_hand_only_preview_script_does_not_require_arm_mode(self):
        args = parse_only("scripts/replay_episode.py", ["--execute", "--yes", "--arm", "none"])
        self.assertEqual(args.arm, "none")


if __name__ == "__main__":
    unittest.main()
