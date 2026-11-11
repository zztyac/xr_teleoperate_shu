"""Per-episode collection metadata and the x (discard) key, without robot, cameras or XR."""

import argparse
import ast
import datetime
import json
import logging
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "teleop/teleop_hand_and_arm.py"


def load_definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names or
             (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        namespace = dict(os=os, np=np, json=json, datetime=datetime, time=time, Queue=Queue, Empty=Empty,
                         Thread=Thread, logger_mp=Mock(spec=logging.Logger),
                         cv2=SimpleNamespace(imwrite=lambda path, image: Path(path).write_bytes(b"jpg") or True))
        self.Writer = load_definitions(ROOT / "teleop/utils/episode_writer.py", {"EpisodeWriter"}, namespace)["EpisodeWriter"]
        self.task_dir = Path(self.temporary.name) / "task"
        self.image = np.zeros((4, 8, 3), dtype=np.uint8)

    def writer(self, **kwargs):
        writer = self.Writer(str(self.task_dir), rerun_log=False, **kwargs)
        self.addCleanup(lambda: writer.close() if writer.worker_thread.is_alive() else None)
        return writer

    def record(self, writer):
        self.assertTrue(writer.create_episode())
        writer.add_item(colors={"color_0": self.image}, states={"qpos": [0.0]}, actions={})
        return Path(writer.episode_dir).name

    def test_collection_meta_is_written_into_each_episode(self):
        meta = {"schema": 1, "mode": "motion", "keyboard": {"linear_speed": 0.05}}
        writer = self.writer(collection_meta=meta)
        self.record(writer)
        writer.save_episode()
        self.assertTrue(wait_until(writer.is_ready))
        payload = json.loads(Path(writer.json_path).read_text())
        self.assertEqual(payload["info"]["collection"], meta)
        self.assertIn("image", payload["info"])  # existing schema untouched

    def test_no_meta_keeps_old_info(self):
        writer = self.writer()
        self.record(writer)
        writer.save_episode()
        self.assertTrue(wait_until(writer.is_ready))
        self.assertNotIn("collection", json.loads(Path(writer.json_path).read_text())["info"])

    def test_operator_discard_quarantines_and_reuses_the_number(self):
        writer = self.writer()
        first = self.record(writer)
        writer.abort_episode("operator discarded the episode (x)", by_operator=True)
        self.assertTrue(wait_until(writer.is_ready))
        self.assertFalse((self.task_dir / first).exists())
        moved = Path(writer.last_aborted_episode_dir)
        self.assertEqual(moved.parent.name, "task_invalid")
        payload = json.loads((moved / "data.json").read_text())
        self.assertEqual(payload["status"], "aborted")
        self.assertIn("operator", payload["failure_reason"])
        self.assertEqual(self.record(writer), first)  # same episode number again

    def test_camera_abort_keeps_numbering_unchanged(self):
        writer = self.writer()
        first = self.record(writer)
        writer.abort_episode("camera stalled")
        self.assertTrue(wait_until(writer.is_ready))
        self.assertNotEqual(self.record(writer), first)


class MetaTests(unittest.TestCase):
    def setUp(self):
        self.code = load_definitions(MAIN, {"git_state", "build_collection_meta"}, {})

    def args(self, **overrides):
        values = dict(motion=True, arm="G1_29", ee="brainco", input_mode="controller", frequency=30.0,
                      initial_pose_task_name="vehicle_physical_button_press_0903", hold_seated_legs=True,
                      keyboard_arm="right", keyboard_linear_speed=0.05, keyboard_angular_speed=0.5,
                      keyboard_linear_accel=0.15, keyboard_angular_accel=1.5, keyboard_release_gap=0.12,
                      keyboard_repeat_delay=0.75, keyboard_ik_reference="initial", smooth_joint_targets=True)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_motion_meta_records_measured_start_pose(self):
        meta = self.code["build_collection_meta"](
            self.args(), np.arange(14) * 0.1, False, str(ROOT), ["--motion", "--record"])
        json.dumps(meta)  # must be JSON-serializable
        self.assertEqual(meta["mode"], "motion")
        self.assertEqual(meta["initial_pose"], "measured_at_start")
        self.assertEqual(len(meta["start_arm_q"]), 14)
        self.assertIsNone(meta["hold_seated_legs"])
        self.assertEqual(meta["keyboard"]["linear_speed"], 0.05)
        self.assertTrue(meta["smooth_joint_targets"])
        self.assertEqual(meta["argv"], ["--motion", "--record"])

    def test_debug_meta_records_pose_name_and_seated_legs(self):
        meta = self.code["build_collection_meta"](
            self.args(motion=False, smooth_joint_targets=False), np.zeros(14), True, str(ROOT), [])
        self.assertEqual(meta["mode"], "debug")
        self.assertEqual(meta["initial_pose"], "vehicle_physical_button_press_0903")
        self.assertTrue(meta["control_full_body"])
        self.assertTrue(meta["hold_seated_legs"])

    def test_git_state_of_this_repo(self):
        state = self.code["git_state"](str(ROOT))
        self.assertRegex(state["commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(state["branch"])
        self.assertIsInstance(state["dirty"], bool)

    def test_git_state_never_raises(self):
        self.assertIn("error", self.code["git_state"]("/nonexistent/repo"))


class DiscardKeyWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = MAIN.read_text()

    def test_x_is_not_a_motion_key(self):
        code = load_definitions(MAIN, {"KEYBOARD_TRANSLATION_KEYS", "KEYBOARD_ROTATION_KEYS",
                                       "KEYBOARD_RETURN_HOME_KEYS", "KEYBOARD_EE_ACTION_KEY",
                                       "KEYBOARD_CONTROL_KEYS"}, {"np": np})
        self.assertNotIn("x", code["KEYBOARD_CONTROL_KEYS"])

    def test_x_only_requests_discard_while_recording(self):
        on_press = self.source[self.source.index("def on_press(key):"):self.source.index("def on_release(key):")]
        self.assertRegex(on_press, r"elif key == 'x':\s+if RECORD_RUNNING:\s+DISCARD_REQUEST = True")

    def test_loop_discards_before_handling_enter(self):
        discard = self.source.index("if args.record and DISCARD_REQUEST:")
        toggle = self.source.index("if args.record and RECORD_TOGGLE:")
        self.assertLess(discard, toggle)
        block = self.source[discard:toggle]
        self.assertIn("RECORD_RUNNING = False", block)
        self.assertIn("by_operator=True", block)

    def test_writer_receives_collection_meta(self):
        call = self.source.index("recorder = EpisodeWriter(")
        self.assertIn("collection_meta = collection_meta)", self.source[call:call + 800])


if __name__ == "__main__":
    unittest.main()
