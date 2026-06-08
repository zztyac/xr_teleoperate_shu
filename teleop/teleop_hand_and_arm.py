"""车内物理按键数据采集
# UNITREE：
1. 启动灵巧手
sudo ~/brainco_hand_service/bin/brainco_hand_server
2. 启动相机server:
conda activate teleimager
python -m teleimager.image_server --rs

# PC 端：
conda activate tv
cd /home/ubuntu/zty/xr_teleoperate_shu
1. 验证摄像头正常连接：python -m teleimager.image_client --host 192.168.123.164
2. 启动遥操作和数据录制：
  运控模式： python teleop/teleop_hand_and_arm.py --record --motion --task-name vehicle_physical_button_press
  debug模式： python teleop/teleop_hand_and_arm.py --record --task-name vehicle_physical_button_press_ccw --headless
  debug只控制手臂： python teleop/teleop_hand_and_arm.py --record --task-name vehicle_physical_button_press_ccw --debug-arms-only


初始姿态固定从 teleop/initial_target_poses.json 读取，使用 --task-name 对应 JSON key。

运行时按键：
  r      开始机器人跟随键盘 / XR 运动
  q      停止并退出程序
  enter  开始录制当前 episode，或保存正在录制的 episode
  n / p  录制前切换到下一个 / 上一个按键子任务
  1-6    录制前直接选择对应的按键子任务
  w / s  末端沿机器人 X 轴正 / 负方向持续移动
  a / d  末端沿机器人 Y 轴正 / 负方向持续移动
  e / c  末端沿机器人 Z 轴正 / 负方向持续移动
  u / o  末端绕机器人 X 轴负 / 正方向持续旋转
  i / k  末端绕机器人 Y 轴正 / 负方向持续旋转
  j / l  末端绕机器人 Z 轴正 / 负方向持续旋转
  space  按一下当前键盘控制末端回到初始姿态，手/夹爪恢复张开
  f      按一次切换当前键盘控制末端的食指/夹爪按压状态

每个 episode 会把当前选择的子任务写入 data.json -> text。
子任务语言从 teleop/button_subtasks.json 按 --task-name 读取；未配置的任务使用命令行默认 task 文本。
当前物理按键任务的子任务顺序如下，goal 保持英文，作为训练用语言标签：
  1. front_windshield_defrost
     goal: Press the front windshield defrost button once.
  2. ac_temperature_down_driver
     goal: Press the driver air conditioning temperature down button once.
  3. fan_speed_down
     goal: Press the fan speed down button once.
  4. ac_temperature_down_passenger
     goal: Press the passenger air conditioning temperature down button once.
  5. trunk_open_long_press
     goal: Long-press the trunk open button until the trunk starts opening.
  6. sunshade_open_long_press
     goal: Long-press the sunshade open button until the sunshade starts opening.

推荐采集流程：
  1. 按 r 开始机器人跟随。
  2. 按 1，再按 enter 录制前窗除雾；录完后再按 enter 保存。
  3. 按 2，再按 enter 录制主驾驶空调温度降低；录完后再按 enter 保存。
  4. 按 3，再按 enter 录制空调风速降低；录完后再按 enter 保存。
  5. 按 4，再按 enter 录制副驾驶空调温度降低；录完后再按 enter 保存。
  6. 按 5，再按 enter 录制长按打开后备箱；录完后再按 enter 保存。
  7. 按 6，再按 enter 录制长按打开遮阳板；录完后再按 enter 保存。

注意：
  - 当前子任务会在 create_episode() 前写入 recorder.text。
  - 录制过程中禁止切换子任务；需要先按 enter 保存当前 episode。
  - 数据保存路径为 <task-dir>/vehicle_physical_button_press/episode_xxxx/data.json。
"""


import time
import argparse
import json
import numpy as np
import pinocchio as pin
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
INITIAL_TARGET_Q_CONFIG_PATH = os.path.join(current_dir, "initial_target_poses.json")
BUTTON_SUBTASKS_CONFIG_PATH = os.path.join(current_dir, "button_subtasks.json")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_23_ArmController,
    H1_2_ArmController,
    H1_ArmController,
    H2_ArmController,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    G1_23_JointArmIndex,
    G1_23_JointIndex,
    H1_2_JointArmIndex,
    H1_2_JointIndex,
    H1_JointArmIndex,
    H1_JointIndex,
    H2_JointArmIndex,
    H2_JointIndex,
)
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# 全局状态由键盘/IPC 回调线程和主循环共同读写；保持含义简单，避免在回调里做重逻辑。
# READY 表示当前阶段允许按键触发下一步，RECORD_TOGGLE 只作为一次性边沿信号使用。
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
CURRENT_BUTTON_SUBTASK_IDX = 0
PRESSED_KEYS = set()
PRESSED_KEYS_LOCK = threading.Lock()
KEYBOARD_EE_ACTION_ACTIVE = False
KEYBOARD_RETURN_HOME_ACTIVE = False
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.
# 子任务语言运行时从 button_subtasks.json 加载；列表顺序决定数字键选择顺序。
# 为空表示当前 task-name 没有配置子任务语言，录制时保留命令行 task 文本。
BUTTON_SUBTASKS = []

KEYBOARD_TRANSLATION_KEYS = {
    "w": np.array([1.0, 0.0, 0.0]),
    "s": np.array([-1.0, 0.0, 0.0]),
    "a": np.array([0.0, 1.0, 0.0]),
    "d": np.array([0.0, -1.0, 0.0]),
    "e": np.array([0.0, 0.0, 1.0]),
    "c": np.array([0.0, 0.0, -1.0]),
}
KEYBOARD_ROTATION_KEYS = {
    "u": np.array([-1.0, 0.0, 0.0]),
    "o": np.array([1.0, 0.0, 0.0]),
    "i": np.array([0.0, 1.0, 0.0]),
    "k": np.array([0.0, -1.0, 0.0]),
    "j": np.array([0.0, 0.0, 1.0]),
    "l": np.array([0.0, 0.0, -1.0]),
}
KEYBOARD_EE_ACTION_KEY = "f"
KEYBOARD_RETURN_HOME_KEYS = {"space", " "}
KEYBOARD_CONTROL_KEYS = (
    set(KEYBOARD_TRANSLATION_KEYS)
    | set(KEYBOARD_ROTATION_KEYS)
    | KEYBOARD_RETURN_HOME_KEYS
    | {KEYBOARD_EE_ACTION_KEY}
)


def normalize_key(key) -> str:
    return str(key).lower()


def remember_pressed_key(key) -> bool:
    key = normalize_key(key)
    if key not in KEYBOARD_CONTROL_KEYS:
        return False
    with PRESSED_KEYS_LOCK:
        PRESSED_KEYS.add(key)
    return True


def forget_pressed_key(key) -> bool:
    key = normalize_key(key)
    if key not in KEYBOARD_CONTROL_KEYS:
        return False
    with PRESSED_KEYS_LOCK:
        PRESSED_KEYS.discard(key)
    return True


def get_pressed_keys_snapshot():
    with PRESSED_KEYS_LOCK:
        return set(PRESSED_KEYS)


def clear_pressed_keys():
    global KEYBOARD_EE_ACTION_ACTIVE, KEYBOARD_RETURN_HOME_ACTIVE
    with PRESSED_KEYS_LOCK:
        PRESSED_KEYS.clear()
        KEYBOARD_EE_ACTION_ACTIVE = False
        KEYBOARD_RETURN_HOME_ACTIVE = False


def start_keyboard_return_home():
    global KEYBOARD_EE_ACTION_ACTIVE, KEYBOARD_RETURN_HOME_ACTIVE
    with PRESSED_KEYS_LOCK:
        KEYBOARD_EE_ACTION_ACTIVE = False
        KEYBOARD_RETURN_HOME_ACTIVE = True
        for key in KEYBOARD_RETURN_HOME_KEYS:
            PRESSED_KEYS.discard(key)


def is_keyboard_return_home_active():
    with PRESSED_KEYS_LOCK:
        return KEYBOARD_RETURN_HOME_ACTIVE


def stop_keyboard_return_home():
    global KEYBOARD_RETURN_HOME_ACTIVE
    with PRESSED_KEYS_LOCK:
        KEYBOARD_RETURN_HOME_ACTIVE = False


def toggle_keyboard_ee_action():
    global KEYBOARD_EE_ACTION_ACTIVE
    with PRESSED_KEYS_LOCK:
        if KEYBOARD_EE_ACTION_KEY in PRESSED_KEYS:
            return KEYBOARD_EE_ACTION_ACTIVE, False
        PRESSED_KEYS.add(KEYBOARD_EE_ACTION_KEY)
        KEYBOARD_EE_ACTION_ACTIVE = not KEYBOARD_EE_ACTION_ACTIVE
        return KEYBOARD_EE_ACTION_ACTIVE, True


def is_keyboard_ee_action_active():
    with PRESSED_KEYS_LOCK:
        return KEYBOARD_EE_ACTION_ACTIVE


def pose_matrix_to_list(pose):
    return np.asarray(pose, dtype=float).reshape(4, 4).tolist()


def get_end_effector_poses_from_q(arm_ik, q):
    # 用当前/初始关节角做一次正运动学，得到左右腕部 4x4 位姿矩阵。
    # 键盘控制和录制字段都依赖这个矩阵，顺序必须与 IK 模型保持一致。
    q = np.asarray(q, dtype=float).reshape(-1)
    pin.framesForwardKinematics(arm_ik.reduced_robot.model, arm_ik.reduced_robot.data, q)
    pin.updateFramePlacements(arm_ik.reduced_robot.model, arm_ik.reduced_robot.data)
    left_pose = arm_ik.reduced_robot.data.oMf[arm_ik.L_hand_id].homogeneous.copy()
    right_pose = arm_ik.reduced_robot.data.oMf[arm_ik.R_hand_id].homogeneous.copy()
    return left_pose, right_pose


def get_initial_end_effector_poses_from_q(arm_ik, initial_q):
    return get_end_effector_poses_from_q(arm_ik, initial_q)


def rotation_matrix_from_rotvec(rotvec):
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.eye(3)

    axis = rotvec / angle
    x, y, z = axis
    skew = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotvec_from_rotation_matrix(rotation):
    trace = np.trace(rotation)
    angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)

    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        axis[0] = np.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = np.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = np.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return np.zeros(3)
    return axis / axis_norm * angle


def orthonormalize_rotation(rotation):
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


class KeyboardEndEffectorController:
    def __init__(
        self,
        control_arm="right",
        linear_speed=0.025,
        angular_speed=0.175,
        initial_left_target=None,
        initial_right_target=None,
    ):
        self.control_arm = control_arm
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.initial_left_target = (
            None if initial_left_target is None else np.array(initial_left_target, dtype=float, copy=True)
        )
        self.initial_right_target = (
            None if initial_right_target is None else np.array(initial_right_target, dtype=float, copy=True)
        )
        self.left_target = None if self.initial_left_target is None else self.initial_left_target.copy()
        self.right_target = None if self.initial_right_target is None else self.initial_right_target.copy()

    def _ensure_targets(self, left_wrist_pose, right_wrist_pose):
        if self.left_target is None:
            self.left_target = np.array(left_wrist_pose, dtype=float, copy=True)
            self.initial_left_target = self.left_target.copy()
        if self.right_target is None:
            self.right_target = np.array(right_wrist_pose, dtype=float, copy=True)
            self.initial_right_target = self.right_target.copy()

    def _target_poses(self):
        if self.control_arm == "left":
            return [self.left_target]
        if self.control_arm == "both":
            return [self.left_target, self.right_target]
        return [self.right_target]

    def _target_initial_pose_pairs(self):
        if self.control_arm == "left":
            return [(self.left_target, self.initial_left_target)]
        if self.control_arm == "both":
            return [
                (self.left_target, self.initial_left_target),
                (self.right_target, self.initial_right_target),
            ]
        return [(self.right_target, self.initial_right_target)]

    def _step_pose_toward_initial(self, pose, initial_pose, dt):
        # space 回初始姿态时不直接跳变，而是按线速度/角速度逐步靠近，避免 IK 目标突变。
        translation_error = initial_pose[:3, 3] - pose[:3, 3]
        translation_norm = np.linalg.norm(translation_error)
        translation_step = self.linear_speed * dt
        if translation_norm <= translation_step:
            pose[:3, 3] = initial_pose[:3, 3]
            translation_done = True
        elif translation_norm > 0.0:
            pose[:3, 3] += translation_error / translation_norm * translation_step
            translation_done = False
        else:
            translation_done = True

        rotation_error = initial_pose[:3, :3] @ pose[:3, :3].T
        rotation_vec = rotvec_from_rotation_matrix(rotation_error)
        rotation_angle = np.linalg.norm(rotation_vec)
        rotation_step = self.angular_speed * dt
        if rotation_angle <= rotation_step:
            pose[:3, :3] = initial_pose[:3, :3]
            rotation_done = True
        elif rotation_angle > 0.0:
            step_vec = rotation_vec / rotation_angle * rotation_step
            pose[:3, :3] = orthonormalize_rotation(
                rotation_matrix_from_rotvec(step_vec) @ pose[:3, :3]
            )
            rotation_done = False
        else:
            rotation_done = True

        return translation_done and rotation_done

    def update(self, left_wrist_pose, right_wrist_pose, dt, pressed_keys, return_home_active=False):
        self._ensure_targets(left_wrist_pose, right_wrist_pose)

        if return_home_active or KEYBOARD_RETURN_HOME_KEYS & pressed_keys:
            return_home_done = True
            for pose, initial_pose in self._target_initial_pose_pairs():
                return_home_done = self._step_pose_toward_initial(pose, initial_pose, dt) and return_home_done
            return self.left_target.copy(), self.right_target.copy(), return_home_done

        # 多个方向键同时按下时先合成方向，再归一化，避免斜向移动速度变快。
        translation_dir = np.zeros(3)
        for key, direction in KEYBOARD_TRANSLATION_KEYS.items():
            if key in pressed_keys:
                translation_dir += direction
        translation_norm = np.linalg.norm(translation_dir)
        if translation_norm > 1.0:
            translation_dir /= translation_norm

        rotation_dir = np.zeros(3)
        for key, direction in KEYBOARD_ROTATION_KEYS.items():
            if key in pressed_keys:
                rotation_dir += direction
        rotation_norm = np.linalg.norm(rotation_dir)
        if rotation_norm > 1.0:
            rotation_dir /= rotation_norm

        translation_delta = translation_dir * self.linear_speed * dt
        rotation_delta = rotation_matrix_from_rotvec(rotation_dir * self.angular_speed * dt)

        if np.any(translation_delta) or rotation_norm > 0.0:
            for pose in self._target_poses():
                pose[:3, 3] += translation_delta
                if rotation_norm > 0.0:
                    pose[:3, :3] = orthonormalize_rotation(rotation_delta @ pose[:3, :3])

        return self.left_target.copy(), self.right_target.copy(), False

    def get_selected_button_state(self, action_active):
        if self.control_arm == "left":
            return action_active, False
        if self.control_arm == "both":
            return action_active, action_active
        return False, action_active


def load_button_subtasks_from_config(task_name: str) -> list[dict]:
    # 每个 task-name 可以有独立语言；没有配置时返回空列表，不影响普通录制任务。
    if not task_name or not os.path.exists(BUTTON_SUBTASKS_CONFIG_PATH):
        return []
    try:
        with open(BUTTON_SUBTASKS_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse button subtask file {BUTTON_SUBTASKS_CONFIG_PATH}: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"button subtask file {BUTTON_SUBTASKS_CONFIG_PATH} must contain a JSON object.")

    subtasks = config.get(task_name, [])
    if subtasks is None:
        return []
    if not isinstance(subtasks, list):
        raise ValueError(f"button subtasks for task '{task_name}' must be a JSON list.")

    required_fields = {"id", "goal", "desc", "steps"}
    for index, subtask in enumerate(subtasks, start=1):
        if not isinstance(subtask, dict):
            raise ValueError(f"button subtask #{index} for task '{task_name}' must be a JSON object.")
        missing_fields = required_fields - set(subtask)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"button subtask #{index} for task '{task_name}' is missing: {missing}.")
        for field in required_fields:
            if not isinstance(subtask[field], str) or not subtask[field].strip():
                raise ValueError(f"button subtask #{index} field '{field}' must be a non-empty string.")
    return subtasks


def get_current_button_subtask() -> dict | None:
    if not BUTTON_SUBTASKS:
        return None
    return BUTTON_SUBTASKS[CURRENT_BUTTON_SUBTASK_IDX]


def log_current_button_subtask(prefix="Selected"):
    subtask = get_current_button_subtask()
    if subtask is None:
        logger_mp.info(f"{prefix} button subtask: no language configured for this task.")
        return
    logger_mp.info(
        f"{prefix} button subtask [{CURRENT_BUTTON_SUBTASK_IDX + 1}/{len(BUTTON_SUBTASKS)}] "
        f"{subtask['id']}: {subtask['goal']}"
    )


def select_button_subtask(index: int):
    global CURRENT_BUTTON_SUBTASK_IDX
    if not BUTTON_SUBTASKS:
        return
    if RECORD_RUNNING:
        logger_mp.warning("Cannot switch button subtask while recording. Save the current episode first.")
        return
    CURRENT_BUTTON_SUBTASK_IDX = index % len(BUTTON_SUBTASKS)
    log_current_button_subtask()


def on_press(key):
    global STOP, START, RECORD_TOGGLE
    key = normalize_key(key)
    if key in KEYBOARD_RETURN_HOME_KEYS:
        start_keyboard_return_home()
        logger_mp.info("Keyboard return-home started; end-effector press is OFF.")
        return
    if key == KEYBOARD_EE_ACTION_KEY:
        action_active, changed = toggle_keyboard_ee_action()
        if changed:
            logger_mp.info(f"Keyboard end-effector press {'ON' if action_active else 'OFF'}.")
        return
    if remember_pressed_key(key):
        return
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
        clear_pressed_keys()
    elif key in ('enter', 'return', '\n') and START == True:
        RECORD_TOGGLE = True
    elif key == 'n':
        select_button_subtask(CURRENT_BUTTON_SUBTASK_IDX + 1)
    elif key == 'p':
        select_button_subtask(CURRENT_BUTTON_SUBTASK_IDX - 1)
    elif len(key) == 1 and key.isdigit() and key != '0':
        select_button_subtask(int(key) - 1)
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def on_release(key):
    forget_pressed_key(key)


def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    subtask = get_current_button_subtask()
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "BUTTON_SUBTASK_INDEX": CURRENT_BUTTON_SUBTASK_IDX if subtask is not None else -1,
        "BUTTON_SUBTASK_ID": subtask["id"] if subtask is not None else "",
        "BUTTON_SUBTASK_GOAL": subtask["goal"] if subtask is not None else "",
    }

ARM_TARGET_DOF = {
    "G1_29": 14,
    "G1_23": 10,
    "H1_2": 14,
    "H1": 8,
    "H2": 14,
}
ARM_FULL_BODY_DOF = {
    "G1_29": len(G1_29_JointIndex),
    "G1_23": len(G1_23_JointIndex),
    "H1_2": len(H1_2_JointIndex),
    "H1": len(H1_JointIndex),
    "H2": len(H2_JointIndex),
}
ARM_JOINT_INDICES = {
    "G1_29": [joint.value for joint in G1_29_JointArmIndex],
    "G1_23": [joint.value for joint in G1_23_JointArmIndex],
    "H1_2": [joint.value for joint in H1_2_JointArmIndex],
    "H1": [joint.value for joint in H1_JointArmIndex],
    "H2": [joint.value for joint in H2_JointArmIndex],
}
ALL_JOINT_Q_FIELD = "all_joint_q"
DUAL_ARM_Q_FIELD = "dual_arm_q"
LEGACY_INITIAL_TARGET_Q_FIELD = "initial_target_q"
INITIAL_TARGET_Q_FIELDS = {
    ALL_JOINT_Q_FIELD,
    DUAL_ARM_Q_FIELD,
    LEGACY_INITIAL_TARGET_Q_FIELD,
}
INITIAL_TARGET_Q_MISSING = object()


def parse_joint_q(raw_value, expected_dof: int, field_name: str):
    # initial_target_poses.json 里所有关节数组都在这里做长度校验。
    # 长度不匹配时直接报错，比让控制器拿错关节顺序更安全。
    if raw_value is None:
        return None

    if not isinstance(raw_value, list):
        raise ValueError(f"{field_name} in initial_target_poses.json must be a JSON list.")

    values = [float(value) for value in raw_value]

    if len(values) != expected_dof:
        raise ValueError(f"{field_name} must have {expected_dof} values, got {len(values)}.")
    return values


def parse_initial_target_q(raw_value, expected_dof: int):
    return parse_joint_q(raw_value, expected_dof, LEGACY_INITIAL_TARGET_Q_FIELD)


def extract_dual_arm_q_from_all_joint_q(raw_value, arm: str):
    # all_joint_q 是全身关节数组；双臂控制只需要按该机型枚举抽出左右臂关节。
    all_joint_q = parse_joint_q(raw_value, ARM_FULL_BODY_DOF[arm], ALL_JOINT_Q_FIELD)
    return [all_joint_q[index] for index in ARM_JOINT_INDICES[arm]]


def parse_full_body_q_entry(raw_value, arm: str):
    full_body_dof = ARM_FULL_BODY_DOF[arm]

    if isinstance(raw_value, dict):
        if ALL_JOINT_Q_FIELD in raw_value:
            return parse_joint_q(raw_value[ALL_JOINT_Q_FIELD], full_body_dof, ALL_JOINT_Q_FIELD), ALL_JOINT_Q_FIELD
        if LEGACY_INITIAL_TARGET_Q_FIELD in raw_value:
            legacy_value = raw_value[LEGACY_INITIAL_TARGET_Q_FIELD]
            if isinstance(legacy_value, list) and len(legacy_value) == full_body_dof:
                return parse_joint_q(legacy_value, full_body_dof, LEGACY_INITIAL_TARGET_Q_FIELD), LEGACY_INITIAL_TARGET_Q_FIELD
        return None, None

    if isinstance(raw_value, list) and len(raw_value) == full_body_dof:
        return parse_joint_q(raw_value, full_body_dof, "legacy_full_body_list"), "legacy_full_body_list"
    return None, None


def parse_initial_target_pose_entry(raw_value, arm: str):
    # 兼容三种历史格式：
    # 1. 推荐格式 {all_joint_q, dual_arm_q}
    # 2. 旧字段 initial_target_q
    # 3. 直接给数组的旧格式
    expected_dof = ARM_TARGET_DOF[arm]
    full_body_dof = ARM_FULL_BODY_DOF[arm]

    if isinstance(raw_value, dict):
        if ALL_JOINT_Q_FIELD in raw_value:
            return extract_dual_arm_q_from_all_joint_q(raw_value[ALL_JOINT_Q_FIELD], arm), ALL_JOINT_Q_FIELD
        if DUAL_ARM_Q_FIELD in raw_value:
            return parse_joint_q(raw_value[DUAL_ARM_Q_FIELD], expected_dof, DUAL_ARM_Q_FIELD), DUAL_ARM_Q_FIELD
        if LEGACY_INITIAL_TARGET_Q_FIELD in raw_value:
            legacy_value = raw_value[LEGACY_INITIAL_TARGET_Q_FIELD]
            if isinstance(legacy_value, list) and len(legacy_value) == full_body_dof:
                return extract_dual_arm_q_from_all_joint_q(legacy_value, arm), LEGACY_INITIAL_TARGET_Q_FIELD
            return parse_initial_target_q(legacy_value, expected_dof), LEGACY_INITIAL_TARGET_Q_FIELD
        raise ValueError(
            f"initial target pose entry for arm '{arm}' must contain "
            f"'{ALL_JOINT_Q_FIELD}' or '{DUAL_ARM_Q_FIELD}'."
        )

    if isinstance(raw_value, list) and len(raw_value) == full_body_dof:
        return extract_dual_arm_q_from_all_joint_q(raw_value, arm), "legacy_full_body_list"
    return parse_initial_target_q(raw_value, expected_dof), "legacy_dual_arm_list"


def load_initial_target_q_from_config(task_name: str, arm: str):
    # 读取路径固定为 teleop/initial_target_poses.json。
    # JSON 层级优先按 task_name -> arm 查找，找不到才兼容旧的扁平结构。
    config_path = INITIAL_TARGET_Q_CONFIG_PATH
    if not task_name or not os.path.exists(config_path):
        return INITIAL_TARGET_Q_MISSING

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse initial target q file {config_path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"initial target q file {config_path} must contain a JSON object.")

    pose_entry = config.get(task_name)
    if pose_entry is None:
        return INITIAL_TARGET_Q_MISSING
    if isinstance(pose_entry, dict):
        if arm in pose_entry:
            return pose_entry[arm]
        if INITIAL_TARGET_Q_FIELDS & set(pose_entry):
            return pose_entry
        return INITIAL_TARGET_Q_MISSING
    return pose_entry


def resolve_initial_target_q(args):
    raw_value = load_initial_target_q_from_config(args.task_name, args.arm)
    if raw_value is INITIAL_TARGET_Q_MISSING:
        raise ValueError(
            f"initial_target_q for task '{args.task_name}' and arm '{args.arm}' "
            f"was not found in {INITIAL_TARGET_Q_CONFIG_PATH}. "
            f"Add '{ALL_JOINT_Q_FIELD}' to initial_target_poses.json."
        )

    initial_target_q, source_field = parse_initial_target_pose_entry(raw_value, args.arm)
    logger_mp.info(
        f"Loaded {source_field} for task '{args.task_name}' and arm '{args.arm}' "
        f"from {INITIAL_TARGET_Q_CONFIG_PATH}; using {len(initial_target_q)} dual-arm joints for control."
    )
    return initial_target_q


def resolve_initial_full_body_q(args):
    raw_value = load_initial_target_q_from_config(args.task_name, args.arm)
    if raw_value is INITIAL_TARGET_Q_MISSING:
        raise ValueError(
            f"initial full-body q for task '{args.task_name}' and arm '{args.arm}' "
            f"was not found in {INITIAL_TARGET_Q_CONFIG_PATH}. "
            f"Add '{ALL_JOINT_Q_FIELD}' to initial_target_poses.json."
        )

    initial_full_body_q, source_field = parse_full_body_q_entry(raw_value, args.arm)
    if initial_full_body_q is None:
        raise ValueError(
            f"--debug-full-body requires '{ALL_JOINT_Q_FIELD}' with "
            f"{ARM_FULL_BODY_DOF[args.arm]} values for task '{args.task_name}' and arm '{args.arm}'."
        )

    logger_mp.info(
        f"Loaded {source_field} for task '{args.task_name}' and arm '{args.arm}' "
        f"from {INITIAL_TARGET_Q_CONFIG_PATH}; using {len(initial_full_body_q)} full-body joints in debug mode."
    )
    return initial_full_body_q


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='controller', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--keyboard-arm', type=str, choices=['left', 'right', 'both'], default='right',
                        help='Arm end-effector controlled by keyboard in controller input mode.')
    parser.add_argument('--keyboard-linear-speed', type=float, default=0.025,
                        help='Keyboard end-effector translation speed in m/s.')
    parser.add_argument('--keyboard-angular-speed', type=float, default=0.35,
                        help='Keyboard end-effector rotation speed in rad/s.')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], default='brainco', help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--debug-full-body', action='store_true',
                        help='In debug mode, explicitly control G1_29 legs and waist from all_joint_q while arms remain IK-controlled.')
    parser.add_argument('--debug-arms-only', action='store_true',
                        help='In debug mode, keep the old behavior and control only arms.')
    parser.add_argument('--debug-full-body-velocity-limit', type=float, default=0.5,
                        help='Legacy velocity limit in rad/s for debug full-body non-arm joints.')
    parser.add_argument('--debug-full-body-ramp-duration', type=float, default=6.0,
                        help='Seconds used to ramp G1_29 leg and waist joints to all_joint_q in debug full-body mode.')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = '/mnt/data/zty/json_data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'vehicle_physical_button_press', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'Press in-car physical buttons with the BrainCo dexterous hand.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'Collect 30 FPS demonstrations for in-car physical button press testing.', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: move the BrainCo dexterous hand to the target button; step2: align the fingertip with the button surface; step3: press the button; step4: release and return to a safe pose;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    if args.debug_full_body and args.debug_arms_only:
        parser.error("--debug-full-body and --debug-arms-only cannot be used together.")
    if args.debug_full_body and args.motion:
        parser.error("--debug-full-body can only be used in debug mode. Remove --motion.")
    if args.debug_full_body and args.arm != "G1_29":
        parser.error("--debug-full-body is currently implemented only for --arm G1_29.")
    if args.debug_full_body_velocity_limit <= 0:
        parser.error("--debug-full-body-velocity-limit must be greater than 0.")
    if args.debug_full_body_ramp_duration < 0:
        parser.error("--debug-full-body-ramp-duration must be greater than or equal to 0.")

    try:
        initial_target_q = resolve_initial_target_q(args)
        control_debug_full_body = False
        initial_full_body_q = None
        auto_debug_full_body = (not args.motion and args.arm == "G1_29" and not args.debug_arms_only)
        if args.debug_full_body or auto_debug_full_body:
            try:
                initial_full_body_q = resolve_initial_full_body_q(args)
                control_debug_full_body = True
            except ValueError:
                if args.debug_full_body:
                    raise
                logger_mp.warning(
                    "Debug full-body control is not enabled because this task has no 35-value all_joint_q."
                )
    except ValueError as e:
        parser.error(str(e))
    try:
        BUTTON_SUBTASKS = load_button_subtasks_from_config(args.task_name)
    except ValueError as e:
        parser.error(str(e))
    # 只有当前 task-name 在 button_subtasks.json 中配置了语言，才启用子任务选择和文本覆盖。
    button_subtask_mode = args.record and bool(BUTTON_SUBTASKS)
    logger_mp.debug(f"args: {args}")

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "on_release": on_release, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        right_side_camera_enabled = camera_config.get('right_side_camera', {}).get('enable_zmq', False)
        left_wrist_camera_enabled = camera_config.get('left_wrist_camera', {}).get('enable_zmq', False)
        right_wrist_camera_enabled = camera_config.get('right_wrist_camera', {}).get('enable_zmq', False)
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
            debug_mode_name = result.get('name') if isinstance(result, dict) else None
            if control_debug_full_body and (status != 0 or result is None or debug_mode_name):
                raise RuntimeError(
                    "Failed to enter Unitree debug mode. Full-body low-level control cannot drive legs "
                    "while a high-level motion mode is still active."
                )

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(
                motion_mode=args.motion,
                simulation_mode=args.sim,
                initial_target_q=initial_target_q,
                initial_full_body_q=initial_full_body_q,
                control_full_body=control_debug_full_body,
                body_velocity_limit=args.debug_full_body_velocity_limit,
                body_ramp_duration=args.debug_full_body_ramp_duration,
            )
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim, initial_target_q=initial_target_q)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim, initial_target_q=initial_target_q)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim, initial_target_q=initial_target_q)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim, initial_target_q=initial_target_q)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            left_brainco_index_button = Value('b', False, lock = True)
            right_brainco_index_button = Value('b', False, lock = True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                           control_mode=args.input_mode,
                                           left_index_button=left_brainco_index_button,
                                           right_index_button=right_brainco_index_button)
        else:
            pass

        keyboard_ee_controller = None
        keyboard_last_update_time = time.time()
        if args.input_mode == "controller" and not args.ipc:
            keyboard_initial_left_pose = None
            keyboard_initial_right_pose = None
            try:
                # 键盘末端控制的“home”来自 initial_target_q 正运动学结果；
                # 如果初始姿态配置有误，就退回第一帧 XR 腕部位姿，避免程序直接不可用。
                keyboard_initial_left_pose, keyboard_initial_right_pose = get_initial_end_effector_poses_from_q(
                    arm_ik,
                    arm_ctrl.initial_target_q,
                )
                logger_mp.info("Keyboard end-effector initial pose is set from initial_target_q.")
            except Exception as e:
                logger_mp.warning(
                    f"Failed to get keyboard end-effector initial pose from initial_target_q: {e}. "
                    "Falling back to the first received wrist pose."
                )
            keyboard_ee_controller = KeyboardEndEffectorController(
                control_arm=args.keyboard_arm,
                linear_speed=args.keyboard_linear_speed,
                angular_speed=args.keyboard_angular_speed,
                initial_left_target=keyboard_initial_left_pose,
                initial_right_target=keyboard_initial_right_pose,
            )
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if keyboard_ee_controller is not None:
            logger_mp.info(
                f"⌨️  Keyboard end-effector control is enabled for [{args.keyboard_arm}] arm(s): "
                "move [w/s]=X, [a/d]=Y, [e/c]=Z; rotate [u/o]=Rx-/Rx+, [i/k]=Ry, [j/l]=Rz; "
                "tap [space] to return home and open hand; tap [f] to toggle press."
            )
        if args.record:
            logger_mp.info("🟡  Press [enter] to START or SAVE recording (toggle cycle).")
            if button_subtask_mode:
                logger_mp.info(
                    f"🟡  Press [n]/[p] to switch button subtask, or [1]-[{len(BUTTON_SUBTASKS)}] "
                    "to select it directly before recording."
                )
                log_current_button_subtask(prefix="Initial")
            else:
                logger_mp.info(
                    f"🟡  No subtask language configured for task '{args.task_name}'; "
                    "using --task-goal/--task-desc/--task-steps."
                )
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        keyboard_last_update_time = time.time()

        head_img = None
        right_side_img = None
        left_wrist_img = None
        right_wrist_img = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if right_side_camera_enabled:
                if args.record:
                    right_side_img = img_client.get_right_side_frame()
            if left_wrist_camera_enabled:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if right_wrist_camera_enabled:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # enter 只切换录制状态；真正的 episode 文本标签在 create_episode() 前写入。
            # 这样每段数据都固定绑定开始录制时选中的子任务，录制中禁止切换。
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if button_subtask_mode:
                        subtask = get_current_button_subtask()
                        recorder.text = {
                            "goal": subtask["goal"],
                            "desc": subtask["desc"],
                            "steps": subtask["steps"],
                        }
                        log_current_button_subtask(prefix="Recording")
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            pressed_keys = get_pressed_keys_snapshot()
            left_keyboard_ee_button = False
            right_keyboard_ee_button = False
            if keyboard_ee_controller is not None:
                keyboard_ee_action_active = is_keyboard_ee_action_active()
                left_keyboard_ee_button, right_keyboard_ee_button = (
                    keyboard_ee_controller.get_selected_button_state(keyboard_ee_action_active)
                )
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_brainco_index_button.get_lock():
                    left_brainco_index_button.value = (
                        left_keyboard_ee_button
                        if keyboard_ee_controller is not None
                        else tele_data.left_ctrl_bButton
                    )
                with right_brainco_index_button.get_lock():
                    right_brainco_index_button.value = (
                        right_keyboard_ee_button
                        if keyboard_ee_controller is not None
                        else tele_data.right_ctrl_bButton
                    )
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = (
                        (0.0 if left_keyboard_ee_button else 10.0)
                        if keyboard_ee_controller is not None
                        else tele_data.left_ctrl_triggerValue
                    )
                with right_gripper_value.get_lock():
                    right_gripper_value.value = (
                        (0.0 if right_keyboard_ee_button else 10.0)
                        if keyboard_ee_controller is not None
                        else tele_data.right_ctrl_triggerValue
                    )
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion and keyboard_ee_controller is None:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            left_wrist_pose = tele_data.left_wrist_pose
            right_wrist_pose = tele_data.right_wrist_pose
            if keyboard_ee_controller is not None:
                keyboard_dt = min(max(start_time - keyboard_last_update_time, 0.0), 0.1)
                keyboard_last_update_time = start_time
                keyboard_return_home_active = is_keyboard_return_home_active()
                left_wrist_pose, right_wrist_pose, return_home_done = keyboard_ee_controller.update(
                    tele_data.left_wrist_pose,
                    tele_data.right_wrist_pose,
                    keyboard_dt,
                    pressed_keys,
                    return_home_active=keyboard_return_home_active,
                )
                if keyboard_return_home_active and return_home_done:
                    stop_keyboard_return_home()
                    logger_mp.info("Keyboard return-home reached initial pose.")
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist_pose, right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [0.0, 0.0, 0.0] if keyboard_ee_controller is not None else [
                            -tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                            -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                            -tele_data.right_ctrl_thumbstickValue[0] * 0.3,
                        ]
                elif args.ee == "brainco" and args.input_mode == "controller":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [0.0, 0.0, 0.0] if keyboard_ee_controller is not None else [
                            -tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                            -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                            -tele_data.right_ctrl_thumbstickValue[0] * 0.3,
                        ]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # state 记录机器人实际状态，action 记录本轮下发/期望动作；
                # 后续训练会依赖二者的时间对齐，不要在这里重排字段含义。
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                current_left_wrist_pose, current_right_wrist_pose = get_end_effector_poses_from_q(
                    arm_ik,
                    current_lr_arm_q,
                )
                left_arm_ee_state_pose = pose_matrix_to_list(current_left_wrist_pose)
                right_arm_ee_state_pose = pose_matrix_to_list(current_right_wrist_pose)
                left_arm_ee_action_pose = pose_matrix_to_list(left_wrist_pose)
                right_arm_ee_action_pose = pose_matrix_to_list(right_wrist_pose)
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        # 双目头部相机左右半幅分别写入 color_0/color_1；
                        # 后续腕部/侧视相机编号从 color_2 开始，保持数据集通道稳定。
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if left_wrist_camera_enabled:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if right_wrist_camera_enabled:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                        if right_side_camera_enabled:
                            if right_side_img is not None:
                                colors[f"color_{4}"] = right_side_img.bgr
                            else:
                                logger_mp.warning("Right side image is None!")
                    else:
                        # 非双目模式下头部整图占 color_0，腕部/侧视相机编号整体前移一位。
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if left_wrist_camera_enabled:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if right_wrist_camera_enabled:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                        if right_side_camera_enabled:
                            if right_side_img is not None:
                                colors[f"color_{3}"] = right_side_img.bgr
                            else:
                                logger_mp.warning("Right side image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },
                        "left_arm_ee_pose": {
                            "pose": left_arm_ee_state_pose,
                        },
                        "right_arm_ee_pose": {
                            "pose": right_arm_ee_state_pose,
                        },
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },
                        "left_arm_ee_pose": {
                            "pose": left_arm_ee_action_pose,
                        },
                        "right_arm_ee_pose": {
                            "pose": right_arm_ee_action_pose,
                        },
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            # 主循环按 --frequency 控制采集和控制节奏；如果本轮已经超时，则不再额外 sleep。
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        clear_pressed_keys()
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
