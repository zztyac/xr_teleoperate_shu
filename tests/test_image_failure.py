"""Exercise image failures without importing the robot SDK or opening devices."""

import ast
import datetime
import json
import logging
import os
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


def load_definition(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class ImageFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.logger = Mock(spec=logging.Logger)
        self.cv2 = SimpleNamespace(imwrite=self.write_image)
        namespace = dict(os=os, np=np, json=json, datetime=datetime, time=time,
                         Queue=Queue, Empty=Empty, Thread=Thread,
                         cv2=self.cv2, logger_mp=self.logger)
        self.Writer = load_definition("teleop/utils/episode_writer.py", "EpisodeWriter", namespace)
        self.collect = load_definition("teleop/teleop_hand_and_arm.py", "collect_recording_images",
                                       {"EpisodeWriter": self.Writer})
        self.image = np.zeros((4, 8, 3), dtype=np.uint8)

    @staticmethod
    def write_image(path, image):
        Path(path).write_bytes(b"mock encoded image")
        return True

    def new_writer(self):
        writer = self.Writer(str(Path(self.temporary.name) / "data"), rerun_log=False)
        self.addCleanup(self.close_if_running, writer)
        self.assertTrue(writer.create_episode())
        return writer

    @staticmethod
    def close_if_running(writer):
        if writer.worker_thread.is_alive():
            try:
                writer.close()
            except RuntimeError:
                pass

    def test_success_preserves_schema_and_does_not_mutate_input(self):
        writer = self.new_writer()
        colors = {"color_0": self.image}
        writer.add_item(colors=colors, depths={}, states={"qpos": [1]}, actions={})
        writer.close()
        payload = json.loads(Path(writer.json_path).read_text())
        self.assertEqual(payload["data"][0]["states"], {"qpos": [1]})
        self.assertEqual(payload["data"][0]["colors"], {"color_0": "colors/000000_color_0.jpg"})
        self.assertTrue((Path(writer.episode_dir) / payload["data"][0]["colors"]["color_0"]).exists())
        self.assertIs(colors["color_0"], self.image)

    def test_invalid_images_rejected_before_queueing(self):
        writer = self.new_writer()
        for image in (None, np.empty((0, 4, 3)), np.zeros((2, 2, 2)), "bad image"):
            with self.subTest(image_type=type(image).__name__):
                with self.assertRaisesRegex(ValueError, "Invalid recording image"):
                    writer.add_item(colors={"color_0": image})
        self.assertEqual(writer.item_id, -1)
        self.assertTrue(writer.item_data_queue.empty())
        writer.close()

    def check_write_failure(self, raises):
        writer = self.new_writer()
        calls = []

        def fail_second_image(path, image):
            calls.append(path)
            Path(path).write_bytes(b"partial image")
            if len(calls) == 2:
                if raises:
                    raise RuntimeError("encoder failed")
                return False
            return True

        self.cv2.imwrite = fail_second_image
        writer.add_item(colors={"color_0": self.image, "color_1": self.image})
        writer.save_episode()
        writer.item_data_queue.join()
        with self.assertRaisesRegex(RuntimeError, "Failed to save colors.color_1"):
            writer.is_ready()
        with self.assertRaises(RuntimeError):
            writer.close()
        self.assertFalse(writer.worker_thread.is_alive())
        raw = Path(writer.json_path).read_text()
        self.assertNotIn('"idx"', raw)
        self.assertEqual(list(Path(writer.color_dir).iterdir()), [])
        self.assertFalse(any("saved successfully" in str(call) for call in self.logger.info.call_args_list))
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)

    def test_false_image_write_rolls_back_and_reaches_main_thread(self):
        self.check_write_failure(raises=False)

    def test_image_encoder_exception_rolls_back_and_reaches_main_thread(self):
        self.check_write_failure(raises=True)

    def test_finalize_failure_does_not_hang_close(self):
        writer = self.new_writer()
        writer._save_episode = Mock(side_effect=OSError("disk full"))
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            writer.close()
        self.assertFalse(writer.worker_thread.is_alive())

    def test_successful_recording_can_start_next_episode(self):
        writer = self.new_writer()
        writer.add_item(colors={"color_0": self.image})
        writer.save_episode()
        deadline = time.monotonic() + 3
        while not writer.is_ready() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(writer.is_ready())
        self.assertTrue(writer.create_episode())
        writer.add_item(colors={"color_0": self.image})
        writer.close()
        self.assertEqual(len(json.loads(Path(writer.json_path).read_text())["data"]), 1)

    def test_monocular_slots_and_disabled_cameras(self):
        config = {name: {"enable_zmq": True} for name in
                  ("head_camera", "left_wrist_camera", "right_wrist_camera", "right_side_camera")}
        frames = {name: SimpleNamespace(bgr=self.image) for name in config}
        colors, missing = self.collect(config, frames)
        self.assertEqual(set(colors), {"color_0", "color_1", "color_2", "color_3"})
        self.assertEqual(missing, [])
        config["left_wrist_camera"]["enable_zmq"] = False
        frames["left_wrist_camera"] = None
        colors, missing = self.collect(config, frames)
        self.assertEqual(set(colors), {"color_0", "color_2", "color_3"})
        self.assertEqual(missing, [])

    def test_missing_enabled_camera_rejects_the_recording_frame(self):
        config = {"head_camera": {"enable_zmq": True},
                  "right_side_camera": {"enable_zmq": True}}
        for frame in (None, SimpleNamespace(bgr=None), SimpleNamespace(bgr=np.empty((0, 4, 3)))):
            colors, missing = self.collect(config, {"head_camera": SimpleNamespace(bgr=self.image),
                                                    "right_side_camera": frame})
            self.assertEqual(missing, ["right_side_camera"])
            self.assertIn("color_0", colors)

    def test_binocular_split_and_empty_head_frame(self):
        config = {"head_camera": {"enable_zmq": True, "binocular": True},
                  "right_side_camera": {"enable_zmq": True}}
        colors, missing = self.collect(config, {name: SimpleNamespace(bgr=self.image) for name in config})
        self.assertEqual(set(colors), {"color_0", "color_1", "color_4"})
        self.assertEqual(colors["color_0"].shape, (4, 4, 3))
        self.assertEqual(missing, [])
        _, missing = self.collect(config, {"head_camera": SimpleNamespace(bgr=None),
                                           "right_side_camera": SimpleNamespace(bgr=self.image)})
        self.assertEqual(missing, ["head_camera"])
        _, missing = self.collect(config, {"head_camera": SimpleNamespace(bgr=np.zeros((4, 7, 3))),
                                           "right_side_camera": SimpleNamespace(bgr=self.image)})
        self.assertEqual(missing, ["head_camera"])


if __name__ == "__main__":
    unittest.main()
