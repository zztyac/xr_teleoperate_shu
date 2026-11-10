import os
import cv2
import json
import datetime
import numpy as np
import time
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None, frequency=30, image_size=[640, 480], rerun_log = True):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> RerunLogger initializing ok.\n")
        
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir and not episode_dir.endswith('.zip')]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self._worker_error = None
        self._abort_reason = None
        self.last_aborted_episode_dir = None
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        self.raise_if_failed()
        return self.is_available

    def raise_if_failed(self):
        if self._worker_error is not None:
            raise RuntimeError(
                f"Episode recording failed; the episode was not finalized: {self._worker_error}"
            ) from self._worker_error

    @staticmethod
    def is_valid_image(image):
        return (
            isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim in (2, 3)
            and (image.ndim == 2 or image.shape[2] in (1, 3, 4))
        )

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        self.raise_if_failed()
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self._abort_reason = None
        self.item_id = -1
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        self.first_item = True   # Flag to handle commas in JSON array

        if self.rerun_log:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None):
        self.raise_if_failed()
        if self._abort_reason is not None:
            raise RuntimeError('Cannot append to an aborted recording episode.')
        for group_name, images in (("colors", colors), ("depths", depths)):
            for key, image in (images or {}).items():
                if not self.is_valid_image(image):
                    raise ValueError(f"Invalid recording image: {group_name}.{key}")
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'colors': dict(colors) if colors is not None else None,
            'depths': dict(depths) if depths is not None else None,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    if self._worker_error is None and self._abort_reason is None:
                        self._process_item_data(item_data)
                except Exception as e:
                    self._worker_error = e
                    logger_mp.error(f"Episode recording failed (idx={item_data['idx']}): {e}")
                finally:
                    self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Abort is processed by the writer, after any in-flight image write.
            # The control loop only requests it and never waits for disk I/O.
            if self._abort_reason is not None and not self.is_available and self.item_data_queue.empty():
                if self._worker_error is None:
                    try:
                        self._quarantine_aborted_episode()
                    except Exception as e:
                        self._worker_error = e
                        logger_mp.error(f'Failed to quarantine aborted episode: {e}')
            elif self.need_save and self.item_data_queue.empty():
                if self._worker_error is not None:
                    self.need_save = False
                else:
                    try:
                        self._save_episode()
                    except Exception as e:
                        self._worker_error = e
                        self.need_save = False
                        logger_mp.error(f"Failed to finalize episode: {e}")

    def _write_images(self, images, directory, group_name, idx, attempted_paths):
        paths = {}
        for key, image in (images or {}).items():
            if not self.is_valid_image(image):
                raise ValueError(f"Invalid recording image: {group_name}.{key}")
            name = f'{str(idx).zfill(6)}_{key}.jpg'
            path = os.path.join(directory, name)
            attempted_paths.append(path)
            try:
                written = cv2.imwrite(path, image)
            except Exception as e:
                raise OSError(f"Failed to save {group_name}.{key} image: {path}") from e
            if not written:
                raise OSError(f"Failed to save {group_name}.{key} image: {path}")
            paths[key] = os.path.join(group_name, name)
        return paths

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        # Commit image references only after every image in this frame is saved.
        attempted_paths = []
        try:
            color_paths = self._write_images(colors, self.color_dir, 'colors', idx, attempted_paths)
            depth_paths = self._write_images(depths, self.depth_dir, 'depths', idx, attempted_paths)
        except Exception:
            for path in attempted_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError as e:
                    logger_mp.warning(f"Failed to remove incomplete image {path}: {e}")
            raise
        if colors is not None:
            item_data['colors'] = color_paths
        if depths is not None:
            item_data['depths'] = depth_paths

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        current_record_time = time.time()
        logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{current_record_time}")

        # Log data if necessary
        if self.rerun_log:
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.raise_if_failed()
        if self._abort_reason is not None:
            raise RuntimeError('Cannot save an aborted episode as a successful recording.')
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def abort_episode(self, reason):
        """Request asynchronous quarantine; never interrupt the robot loop."""
        self.raise_if_failed()
        if self.is_available or self._abort_reason is not None:
            return
        self.need_save = False
        self._abort_reason = str(reason)
        logger_mp.error(
            f'[采集已中止] {self.episode_dir}: {reason}。本段将移出正常数据目录，'
            '图像恢复后需重新按 Enter 录制。'
        )

    def _quarantine_aborted_episode(self):
        import shutil
        reason = self._abort_reason
        # Preserve partial evidence, explicitly marked invalid, outside task_dir.
        with open(self.json_path, 'a', encoding='utf-8') as f:
            f.write('\n],\n"status": "aborted",\n"failure_reason": ')
            f.write(json.dumps(reason, ensure_ascii=False))
            f.write('\n}')
        quarantine_root = self.task_dir.rstrip(os.sep) + '_invalid'
        os.makedirs(quarantine_root, exist_ok=True)
        destination = os.path.join(
            quarantine_root, f'{os.path.basename(self.episode_dir)}_{time.time_ns()}'
        )
        shutil.move(self.episode_dir, destination)
        self.last_aborted_episode_dir = destination
        self.need_save = False
        self.is_available = True
        logger_mp.error(f'[异常片段已隔离] {destination}；不作为正常示范保存。')

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n]\n}")      # Close the JSON array and object

        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        try:
            self.item_data_queue.join()
            if not self.is_available and self._worker_error is None and self._abort_reason is None:
                self.need_save = True
            while not self.is_available and self._worker_error is None:
                time.sleep(0.01)
        finally:
            self.stop_worker = True
            self.worker_thread.join()
        self.raise_if_failed()
