"""车内物理按键数据采集

使用真实相机时，先启动图像服务：
  python -m teleimager.image_server --rs
  python -m teleimager.image_client --host 192.168.123.164
启动遥操作和数据录制：
motion : python teleop/teleop_hand_and_arm.py --record --motion --task-name vehicle_physical_button_press
debug : python teleop/teleop_hand_and_arm.py --record --task-name vehicle_physical_button_press

车内初始姿态默认从 teleop/initial_target_poses.json 读取。
如需切换姿态，可使用 --initial-target-q-name 指定配置中的姿态名。

运行时按键：
  r      开始机器人跟随 XR 运动
  q      停止并退出程序
  s      开始录制当前 episode，或保存正在录制的 episode
  n / p  录制前切换到下一个 / 上一个按键子任务
  1-5    录制前直接选择对应的按键子任务

每个 episode 会把当前选择的子任务写入 data.json -> text。
子任务顺序如下，goal 保持英文，作为训练用语言标签：
  1. front_windshield_defrost
     goal: Press the front windshield defrost button once.
  2. ac_temperature_down
     goal: Press the air conditioning temperature down button once.
  3. fan_speed_down
     goal: Press the fan speed down button once.
  4. trunk_open_long_press
     goal: Long-press the trunk open button until the trunk starts opening.
  5. sunshade_open_long_press
     goal: Long-press the sunshade open button until the sunshade starts opening.

推荐采集流程：
  1. 按 r 开始机器人跟随。
  2. 按 1，再按 s 录制前窗除雾；录完后再按 s 保存。
  3. 按 2，再按 s 录制空调温度降低；录完后再按 s 保存。
  4. 按 3，再按 s 录制空调风速降低；录完后再按 s 保存。
  5. 按 4，再按 s 录制长按打开后备箱；录完后再按 s 保存。
  6. 按 5，再按 s 录制长按打开遮阳板；录完后再按 s 保存。

注意：
  - 当前子任务会在 create_episode() 前写入 recorder.text。
  - 录制过程中禁止切换子任务；需要先按 s 保存当前 episode。
  - 数据保存路径为 <task-dir>/vehicle_physical_button_press/episode_xxxx/data.json。
"""


import time
import argparse
import ast
import json
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

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController
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

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
CURRENT_BUTTON_SUBTASK_IDX = 0
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
SHORT_PRESS_STEPS = "step1: move the dexterous hand to the target button; step2: align the fingertip with the button surface; step3: press the button once; step4: release the button; step5: return to a safe pose;"
LONG_PRESS_STEPS = "step1: move the dexterous hand to the target button; step2: align the fingertip with the button surface; step3: press and hold the button; step4: keep holding until the target function starts; step5: release the button; step6: return to a safe pose;"

BUTTON_SUBTASKS = [
    {
        "id": "front_windshield_defrost",
        "goal": "Press the front windshield defrost button once.",
        "desc": "Collect a demonstration for the in-car physical button task: front windshield defrost.",
        "steps": SHORT_PRESS_STEPS,
    },
    {
        "id": "ac_temperature_down",
        "goal": "Press the air conditioning temperature down button once.",
        "desc": "Collect a demonstration for the in-car physical button task: air conditioning temperature down.",
        "steps": SHORT_PRESS_STEPS,
    },
    {
        "id": "fan_speed_down",
        "goal": "Press the fan speed down button once.",
        "desc": "Collect a demonstration for the in-car physical button task: fan speed down.",
        "steps": SHORT_PRESS_STEPS,
    },
    {
        "id": "trunk_open_long_press",
        "goal": "Long-press the trunk open button until the trunk starts opening.",
        "desc": "Collect a demonstration for the in-car physical button task: trunk open by long press.",
        "steps": LONG_PRESS_STEPS,
    },
    {
        "id": "sunshade_open_long_press",
        "goal": "Long-press the sunshade open button until the sunshade starts opening.",
        "desc": "Collect a demonstration for the in-car physical button task: sunshade open by long press.",
        "steps": LONG_PRESS_STEPS,
    },
]


def get_current_button_subtask() -> dict:
    return BUTTON_SUBTASKS[CURRENT_BUTTON_SUBTASK_IDX]


def log_current_button_subtask(prefix="Selected"):
    subtask = get_current_button_subtask()
    logger_mp.info(
        f"{prefix} button subtask [{CURRENT_BUTTON_SUBTASK_IDX + 1}/{len(BUTTON_SUBTASKS)}] "
        f"{subtask['id']}: {subtask['goal']}"
    )


def select_button_subtask(index: int):
    global CURRENT_BUTTON_SUBTASK_IDX
    if RECORD_RUNNING:
        logger_mp.warning("Cannot switch button subtask while recording. Save the current episode first.")
        return
    CURRENT_BUTTON_SUBTASK_IDX = index % len(BUTTON_SUBTASKS)
    log_current_button_subtask()


def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    elif key == 'n':
        select_button_subtask(CURRENT_BUTTON_SUBTASK_IDX + 1)
    elif key == 'p':
        select_button_subtask(CURRENT_BUTTON_SUBTASK_IDX - 1)
    elif key in ('1', '2', '3', '4', '5'):
        select_button_subtask(int(key) - 1)
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    subtask = get_current_button_subtask()
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "BUTTON_SUBTASK_INDEX": CURRENT_BUTTON_SUBTASK_IDX,
        "BUTTON_SUBTASK_ID": subtask["id"],
        "BUTTON_SUBTASK_GOAL": subtask["goal"],
    }

ARM_TARGET_DOF = {
    "G1_29": 14,
    "G1_23": 10,
    "H1_2": 14,
    "H1": 8,
    "H2": 14,
}
INITIAL_TARGET_Q_MISSING = object()


def parse_initial_target_q(raw_value, expected_dof: int):
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            parsed = raw_value.split(",")
    else:
        parsed = raw_value

    if isinstance(parsed, (int, float)):
        values = [float(parsed)]
    else:
        values = [float(value) for value in parsed]

    if len(values) != expected_dof:
        raise ValueError(f"initial_target_q for this arm must have {expected_dof} values, got {len(values)}.")
    return values


def load_named_initial_target_q(config_path: str, pose_name: str, arm: str):
    if not pose_name or not os.path.exists(config_path):
        return INITIAL_TARGET_Q_MISSING

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse initial target q file {config_path}: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"initial target q file {config_path} must contain a JSON object.")

    pose_entry = config.get(pose_name)
    if pose_entry is None:
        return INITIAL_TARGET_Q_MISSING
    if isinstance(pose_entry, dict):
        return pose_entry.get(arm, INITIAL_TARGET_Q_MISSING)
    return pose_entry


def resolve_initial_target_q(args):
    expected_dof = ARM_TARGET_DOF[args.arm]
    if args.initial_target_q is not None:
        return parse_initial_target_q(args.initial_target_q, expected_dof)

    pose_name = args.initial_target_q_name
    if pose_name is None and args.record:
        pose_name = args.task_name

    raw_value = load_named_initial_target_q(args.initial_target_q_file, pose_name, args.arm)
    if raw_value is INITIAL_TARGET_Q_MISSING:
        if args.initial_target_q_name is not None:
            raise ValueError(
                f"initial target q name '{args.initial_target_q_name}' for arm '{args.arm}' "
                f"was not found in {args.initial_target_q_file}."
            )
        return None

    logger_mp.info(
        f"Loaded initial_target_q '{pose_name}' for {args.arm} from {args.initial_target_q_file}."
    )
    return parse_initial_target_q(raw_value, expected_dof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='controller', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--initial-target-q', '--initial_target_q', dest='initial_target_q', type=str, default=None,
                        help='Initial dual-arm joint target, e.g. "[0, 0, ...]" or "0,0,...". Overrides the named JSON config.')
    parser.add_argument('--initial-target-q-file', '--initial-target-pose-file', dest='initial_target_q_file',
                        type=str, default=INITIAL_TARGET_Q_CONFIG_PATH,
                        help='JSON file containing named initial dual-arm joint targets.')
    parser.add_argument('--initial-target-q-name', '--initial-target-pose', dest='initial_target_q_name',
                        type=str, default=None,
                        help='Named initial target in the JSON file. Default uses --task-name when --record is enabled.')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], default='brainco', help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
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
    try:
        initial_target_q = resolve_initial_target_q(args)
    except ValueError as e:
        parser.error(str(e))
    button_subtask_mode = args.record and args.task_name == 'vehicle_physical_button_press'
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
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_camera_name = 'right_side_camera'
        xr_camera_config = camera_config.get(xr_camera_name)
        if not isinstance(xr_camera_config, dict):
            raise RuntimeError(
                f"VR display camera '{xr_camera_name}' is not configured. "
                "Please enable right_side_camera in the teleimager camera config."
            )
        right_side_camera_enabled = camera_config.get('right_side_camera', {}).get('enable_zmq', False)
        left_wrist_camera_enabled = camera_config.get('left_wrist_camera', {}).get('enable_zmq', False)
        right_wrist_camera_enabled = camera_config.get('right_wrist_camera', {}).get('enable_zmq', False)
        xr_need_local_img = not (args.display_mode == 'pass-through' or xr_camera_config['enable_webrtc'])
        logger_mp.info(f"VR display camera source: {xr_camera_name}")

        # televuer_wrapper: obtain XR pose data and transmit the right-side camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=xr_camera_config['binocular'],
                                     img_shape=xr_camera_config['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=xr_camera_config['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=xr_camera_config['enable_zmq'],
                                     webrtc=xr_camera_config['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{xr_camera_config['webrtc_port']}/offer",
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim, initial_target_q=initial_target_q)
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
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
            if button_subtask_mode:
                logger_mp.info("🟡  Press [n]/[p] to switch button subtask, or [1]-[5] to select it directly before recording.")
                log_current_button_subtask(prefix="Initial")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if right_side_camera_enabled and xr_need_local_img:
                right_side_img = img_client.get_right_side_frame()
                if right_side_img.bgr is not None:
                    tv_wrapper.render_to_xr(right_side_img.bgr)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()

        head_img = None
        right_side_img = None
        left_wrist_img = None
        right_wrist_img = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record:
                    head_img = img_client.get_head_frame()
            if right_side_camera_enabled:
                if args.record or xr_need_local_img:
                    right_side_img = img_client.get_right_side_frame()
                if xr_need_local_img and right_side_img.bgr is not None:
                    tv_wrapper.render_to_xr(right_side_img.bgr)
            if left_wrist_camera_enabled:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if right_wrist_camera_enabled:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
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
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_brainco_index_button.get_lock():
                    left_brainco_index_button.value = tele_data.left_ctrl_bButton
                with right_brainco_index_button.get_lock():
                    right_brainco_index_button.value = tele_data.right_ctrl_bButton
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion:
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
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
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
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif args.ee == "brainco" and args.input_mode == "controller":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
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

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
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
            # 决定控制频率
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
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
