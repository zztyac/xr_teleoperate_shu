#!/usr/bin/env python3
"""
python3 scripts/replay_episode_raw.py --episode-path /mnt/data/zty/json_data/vehicle_physical_button_press_ccw/episode_0334

新增末端位姿回放：
python3 scripts/replay_episode_raw.py \
  --episode-path /mnt/data/zty/json_data/vehicle_physical_button_press_ccw/episode_0060 \
  --endpose \
  --yes

"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE_PATH = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press_ccw/episode_0005" 
)
DEFAULT_INITIAL_POSE_CONFIG = REPO_ROOT / "teleop" / "initial_target_poses.json"
DEFAULT_INITIAL_POSE_TASK = "vehicle_physical_button_press_ccw"
DEFAULT_UNITREE_LEROBOT_ROOT = Path("/home/ubuntu/zty/unitree_lerobot")
ARM_DOF = {"G1_29": 14, "G1_23": 10}
FULL_BODY_DOF = {"G1_29": 35}
ARM_JOINT_INDICES = {
    "G1_29": list(range(15, 29)),
    "G1_23": [15, 16, 17, 18, 19, 22, 23, 24, 25, 26],
}


logger = logging.getLogger("replay_robot_raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a raw Unitree JSON episode."
    )
    parser.add_argument(
        "--arm",
        choices=["G1_29", "G1_23"],
        default="G1_29",
        help="Arm controller type.",
    )
    parser.add_argument(
        "--tau-source",
        choices=["zero", "unitree_lerobot"],
        default="unitree_lerobot",
        help="Forward torque source. Use unitree_lerobot to match replay_robot_lerobot.py.",
    )
    parser.add_argument(
        "--arm-action-source",
        choices=["qpos", "endpose"],
        default="qpos",
        help=(
            "Arm action source. Use qpos for actions.left_arm/right_arm.qpos, "
            "or endpose for actions.left_arm_ee_pose/right_arm_ee_pose.pose via IK."
        ),
    )
    parser.add_argument(
        "--endpose",
        dest="arm_action_source",
        action="store_const",
        const="endpose",
        help="Shortcut for --arm-action-source endpose.",
    )
    parser.add_argument(
        "--ee",
        choices=["none", "brainco"],
        default="brainco",
        help="End-effector controller type.",
    )
    parser.add_argument(
        "--no-ee",
        dest="ee",
        action="store_const",
        const="none",
        help="Do not replay end-effector actions.",
    )
    parser.add_argument(
        "--motion",
        action="store_true",
        help="Enable robot motion mode.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use simulation mode for robot controllers.",
    )
    parser.add_argument(
        "--network-interface",
        type=str,
        default=None,
        help="DDS network interface, for example eth0. Defaults to SDK behavior.",
    )
    parser.add_argument(
        "--skip-debug-mode",
        action="store_true",
        help="Do not call MotionSwitcher.Enter_Debug_Mode() before replay.",
    )
    parser.add_argument(
        "--episode-path",
        type=Path,
        default=DEFAULT_EPISODE_PATH,
        help="Episode directory or data.json path.",
    )
    parser.add_argument(
        "--initial-pose-config",
        type=Path,
        default=DEFAULT_INITIAL_POSE_CONFIG,
        help="initial_target_poses.json used for replay initialization.",
    )
    parser.add_argument(
        "--initial-pose-task",
        type=str,
        default=DEFAULT_INITIAL_POSE_TASK,
        help="Task key inside initial_target_poses.json used for replay initialization.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=30,
        help="Replay frequency in Hz. Defaults to info.image.fps from data.json.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start frame index inside the episode data list.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of replayed frames.",
    )
    parser.add_argument(
        "--init-hold-s",
        type=float,
        default=1.0,
        help="Seconds to hold the first state before replay.",
    )
    parser.add_argument(
        "--skip-full-body-init",
        action="store_true",
        help="Do not use configured all_joint_q to initialize the full body.",
    )
    parser.add_argument(
        "--full-body-init-ramp-s",
        type=float,
        default=6.0,
        help="Seconds used by G1_29 debug full-body control to ramp to configured all_joint_q.",
    )
    parser.add_argument(
        "--full-body-init-velocity-limit",
        type=float,
        default=0.5,
        help="Velocity limit in rad/s for G1_29 non-arm joints during full-body initialization.",
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Do not move to the first recorded state before replay.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive start prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the raw episode without importing robot code.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log one action every N frames. Use 1 to log every frame.",
    )
    args = parser.parse_args()
    if args.full_body_init_ramp_s < 0:
        parser.error("--full-body-init-ramp-s must be greater than or equal to 0.")
    if args.full_body_init_velocity_limit <= 0:
        parser.error("--full-body-init-velocity-limit must be greater than 0.")
    return args


def resolve_data_path(path: Path) -> Path:
    if path.is_dir():
        return path / "data.json"
    return path


def load_episode(path: Path) -> tuple[dict, list[dict]]:
    data_path = resolve_data_path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"data.json not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("data")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{data_path} does not contain a non-empty data list")

    return payload, frames


def parse_joint_array(raw_value, expected_dof: int, field_name: str) -> np.ndarray:
    if not isinstance(raw_value, list):
        raise TypeError(f"{field_name} must be a JSON list")

    q = np.asarray(raw_value, dtype=np.float64).reshape(-1)
    if q.size != expected_dof:
        raise ValueError(f"{field_name} must have {expected_dof} values, got {q.size}")
    return q


def extract_arm_q_from_full_body(full_body_q: np.ndarray, arm_name: str) -> np.ndarray:
    if arm_name not in ARM_JOINT_INDICES:
        raise ValueError(f"Cannot extract arm joints from full-body q for arm {arm_name}")
    indices = ARM_JOINT_INDICES[arm_name]
    if max(indices) >= full_body_q.size:
        raise ValueError(
            f"full-body q has {full_body_q.size} values, cannot extract arm indices {indices}"
        )
    return np.asarray([full_body_q[index] for index in indices], dtype=np.float64)


def load_initial_pose_from_config(
    config_path: Path,
    task_name: str,
    arm_name: str,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"initial pose config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"initial pose config must contain a JSON object: {config_path}")
    if task_name not in config:
        raise KeyError(f"Task '{task_name}' not found in {config_path}")

    entry = config[task_name]
    if isinstance(entry, dict) and arm_name in entry:
        entry = entry[arm_name]

    arm_dof = ARM_DOF[arm_name]
    full_body_q = None
    arm_q = None
    source_parts = []

    if isinstance(entry, dict):
        if "all_joint_q" in entry:
            expected_full_dof = FULL_BODY_DOF.get(arm_name)
            if expected_full_dof is None:
                raise ValueError(f"{arm_name} does not support all_joint_q initialization")
            full_body_q = parse_joint_array(
                entry["all_joint_q"],
                expected_full_dof,
                f"{task_name}.{arm_name}.all_joint_q",
            )
            source_parts.append("all_joint_q")

        if "dual_arm_q" in entry:
            arm_q = parse_joint_array(
                entry["dual_arm_q"],
                arm_dof,
                f"{task_name}.{arm_name}.dual_arm_q",
            )
            source_parts.append("dual_arm_q")
        elif full_body_q is not None:
            arm_q = extract_arm_q_from_full_body(full_body_q, arm_name)
            source_parts.append("all_joint_q->dual_arm_q")
    elif isinstance(entry, list):
        if len(entry) == arm_dof:
            arm_q = parse_joint_array(entry, arm_dof, f"{task_name}.{arm_name}")
            source_parts.append("legacy_dual_arm_list")
        else:
            expected_full_dof = FULL_BODY_DOF.get(arm_name)
            if expected_full_dof is not None and len(entry) == expected_full_dof:
                full_body_q = parse_joint_array(entry, expected_full_dof, f"{task_name}.{arm_name}")
                arm_q = extract_arm_q_from_full_body(full_body_q, arm_name)
                source_parts.append("legacy_full_body_list")

    if arm_q is None:
        raise ValueError(
            f"Initial pose for task '{task_name}' and arm '{arm_name}' must contain "
            "'dual_arm_q' or 'all_joint_q'."
        )

    return arm_q, full_body_q, "+".join(source_parts)


def qpos(frame: dict, section: str, name: str) -> np.ndarray:
    try:
        values = frame[section][name]["qpos"]
    except KeyError as exc:
        idx = frame.get("idx", "?")
        raise KeyError(f"frame {idx} missing {section}.{name}.qpos") from exc

    if not isinstance(values, list):
        idx = frame.get("idx", "?")
        raise TypeError(f"frame {idx} {section}.{name}.qpos is not a list")

    return np.asarray(values, dtype=np.float64)


def optional_qpos(frame: dict, section: str, name: str) -> np.ndarray | None:
    section_data = frame.get(section)
    if not isinstance(section_data, dict):
        return None
    part_data = section_data.get(name)
    if not isinstance(part_data, dict) or "qpos" not in part_data:
        return None

    values = part_data["qpos"]
    if values == []:
        return None
    if not isinstance(values, list):
        idx = frame.get("idx", "?")
        raise TypeError(f"frame {idx} {section}.{name}.qpos is not a list")
    return np.asarray(values, dtype=np.float64)


def arm_qpos(frame: dict, section: str) -> np.ndarray:
    return np.concatenate(
        (qpos(frame, section, "left_arm"), qpos(frame, section, "right_arm"))
    )


def body_qpos(frame: dict, section: str) -> np.ndarray | None:
    return optional_qpos(frame, section, "body")


def has_arm_qpos(frame: dict, section: str) -> bool:
    section_data = frame.get(section)
    if not isinstance(section_data, dict):
        return False
    return all(
        isinstance(section_data.get(name), dict) and "qpos" in section_data[name]
        for name in ("left_arm", "right_arm")
    )


def pose(frame: dict, section: str, name: str) -> np.ndarray:
    try:
        values = frame[section][name]["pose"]
    except KeyError as exc:
        idx = frame.get("idx", "?")
        raise KeyError(f"frame {idx} missing {section}.{name}.pose") from exc

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)

    idx = frame.get("idx", "?")
    raise ValueError(
        f"frame {idx} {section}.{name}.pose must be a 4x4 matrix or 16 values, got shape {arr.shape}"
    )


def arm_endpose(frame: dict, section: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        pose(frame, section, "left_arm_ee_pose"),
        pose(frame, section, "right_arm_ee_pose"),
    )


def has_arm_endpose(frame: dict, section: str) -> bool:
    section_data = frame.get(section)
    if not isinstance(section_data, dict):
        return False
    return all(
        isinstance(section_data.get(name), dict) and "pose" in section_data[name]
        for name in ("left_arm_ee_pose", "right_arm_ee_pose")
    )


def ee_qpos(frame: dict, section: str) -> tuple[np.ndarray, np.ndarray]:
    return qpos(frame, section, "left_ee"), qpos(frame, section, "right_ee")


def infer_frequency(payload: dict, override: float | None) -> float:
    if override is not None:
        frequency = override
    else:
        frequency = payload.get("info", {}).get("image", {}).get("fps", 30)

    frequency = float(frequency)
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    return frequency


def select_frames(
    frames: list[dict], start_index: int, max_frames: int | None
) -> list[dict]:
    if start_index < 0:
        raise ValueError(f"start-index must be >= 0, got {start_index}")
    selected = frames[start_index:]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"max-frames must be positive, got {max_frames}")
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("selected frame range is empty")
    return selected


def validate_frames(
    frames: list[dict],
    require_action_arm_qpos: bool,
    require_action_endpose: bool,
) -> tuple[int | None, int, int, int, bool]:
    first = frames[0]
    arm_dim = arm_qpos(first, "actions").size if has_arm_qpos(first, "actions") else None
    state_arm_dim = arm_qpos(first, "states").size
    left_ee_dim, right_ee_dim = (arr.size for arr in ee_qpos(first, "actions"))
    has_action_endpose = has_arm_endpose(first, "actions")

    if require_action_arm_qpos and arm_dim is None:
        arm_qpos(first, "actions")

    if require_action_endpose or has_action_endpose:
        arm_endpose(first, "actions")
        has_action_endpose = True

    for frame in frames:
        idx = frame.get("idx", "?")
        if arm_dim is not None or require_action_arm_qpos:
            if arm_qpos(frame, "actions").size != arm_dim:
                raise ValueError(f"frame {idx} action arm dimension changed")
        if arm_qpos(frame, "states").size != state_arm_dim:
            raise ValueError(f"frame {idx} state arm dimension changed")
        if has_action_endpose or require_action_endpose:
            left_pose, right_pose = arm_endpose(frame, "actions")
            if left_pose.shape != (4, 4) or right_pose.shape != (4, 4):
                raise ValueError(f"frame {idx} action endpose shape changed")
        left_ee, right_ee = ee_qpos(frame, "actions")
        if left_ee.size != left_ee_dim or right_ee.size != right_ee_dim:
            raise ValueError(f"frame {idx} EE action dimension changed")

    return arm_dim, state_arm_dim, left_ee_dim, right_ee_dim, has_action_endpose


def add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def add_unitree_lerobot_source_path() -> Path:
    candidates = [DEFAULT_UNITREE_LEROBOT_ROOT]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "unitree_lerobot")
        candidates.append(parent.parent / "unitree_lerobot")

    for candidate in candidates:
        if (candidate / "unitree_lerobot" / "eval_robot" / "robot_control" / "robot_arm_ik.py").is_file():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise SystemExit(
        "Could not find unitree_lerobot source. "
        "Expected source path: /home/ubuntu/zty/unitree_lerobot."
    )


def initialize_dds(sim: bool, network_interface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    domain_id = 1 if sim else 0
    if network_interface:
        ChannelFactoryInitialize(domain_id, networkInterface=network_interface)
    else:
        ChannelFactoryInitialize(domain_id)


def enter_debug_mode(require_success: bool = False) -> None:
    from teleop.utils.motion_switcher import MotionSwitcher

    status, result = MotionSwitcher().Enter_Debug_Mode()
    logger.info("Enter debug mode: %s %s", "Success" if status == 0 else "Failed", result)
    if require_success:
        mode_name = result.get("name") if isinstance(result, dict) else None
        if status != 0 or result is None or mode_name:
            raise RuntimeError(
                "Failed to enter Unitree debug mode. Full-body low-level replay cannot "
                f"drive legs while a high-level mode is active: status={status}, result={result}"
            )


def create_arm_controller(
    arm_name: str,
    motion: bool,
    sim: bool,
    initial_target_q: np.ndarray | None,
    initial_full_body_q: np.ndarray | None = None,
    control_full_body: bool = False,
    body_velocity_limit: float = 0.5,
    body_ramp_duration: float = 6.0,
):
    from teleop.robot_control.robot_arm import G1_23_ArmController, G1_29_ArmController

    controllers = {
        "G1_29": G1_29_ArmController,
        "G1_23": G1_23_ArmController,
    }
    if arm_name == "G1_29":
        return G1_29_ArmController(
            motion_mode=motion,
            simulation_mode=sim,
            initial_target_q=initial_target_q,
            initial_full_body_q=initial_full_body_q,
            control_full_body=control_full_body,
            body_velocity_limit=body_velocity_limit,
            body_ramp_duration=body_ramp_duration,
        )

    return controllers[arm_name](
        motion_mode=motion,
        simulation_mode=sim,
        initial_target_q=initial_target_q,
    )


def create_arm_ik(arm_name: str):
    from teleop.robot_control.robot_arm_ik import G1_23_ArmIK, G1_29_ArmIK

    ik_classes = {
        "G1_29": G1_29_ArmIK,
        "G1_23": G1_23_ArmIK,
    }
    return ik_classes[arm_name]()


def create_tau_solver(arm_name: str, tau_source: str, arm_dof: int):
    if tau_source == "zero":
        return lambda _q: np.zeros(arm_dof, dtype=np.float64)

    unitree_lerobot_root = add_unitree_lerobot_source_path()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import (
            G1_23_ArmIK,
            G1_29_ArmIK,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import unitree_lerobot arm IK. "
            "Expected source path: /home/ubuntu/zty/unitree_lerobot."
        ) from exc

    ik_classes = {
        "G1_29": G1_29_ArmIK,
        "G1_23": G1_23_ArmIK,
    }
    old_cwd = Path.cwd()
    try:
        os.chdir(unitree_lerobot_root)
        arm_ik = ik_classes[arm_name]()
    finally:
        os.chdir(old_cwd)

    def solve(q: np.ndarray) -> np.ndarray:
        tau = np.asarray(arm_ik.solve_tau(q), dtype=np.float64).reshape(-1)
        if tau.size != arm_dof:
            raise ValueError(
                f"unitree_lerobot solve_tau returned {tau.size} values, expected {arm_dof}"
            )
        return tau

    return solve


class BraincoDirectController:
    def __init__(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

        self.left_pub = ChannelPublisher("rt/brainco/left/cmd", MotorCmds_)
        self.right_pub = ChannelPublisher("rt/brainco/right/cmd", MotorCmds_)
        self.left_pub.Init()
        self.right_pub.Init()
        self.left_msg = MotorCmds_()
        self.right_msg = MotorCmds_()
        self.left_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        self.right_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        for msg in (self.left_msg, self.right_msg):
            for cmd in msg.cmds:
                cmd.q = 0.0
                cmd.dq = 1.0

    def ctrl_dual_hand(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        left_q = np.clip(np.asarray(left_q, dtype=np.float64).reshape(-1), 0.0, 1.0)
        right_q = np.clip(np.asarray(right_q, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if left_q.size != 6 or right_q.size != 6:
            raise ValueError("BrainCo replay expects 6 q values per hand")
        for idx, value in enumerate(left_q):
            self.left_msg.cmds[idx].q = float(value)
        for idx, value in enumerate(right_q):
            self.right_msg.cmds[idx].q = float(value)
        self.left_pub.Write(self.left_msg)
        self.right_pub.Write(self.right_msg)


def selected_ee_name(args: argparse.Namespace, left_ee_dim: int, right_ee_dim: int) -> str:
    if args.ee == "none":
        return ""
    if left_ee_dim == 0 and right_ee_dim == 0:
        return ""
    return args.ee


def wait_for_start(skip_prompt: bool) -> None:
    if skip_prompt:
        return

    user_input = input(
        "Please enter the start signal (enter 's' to start replay): "
    )
    if user_input.lower() != "s":
        raise SystemExit("Replay cancelled.")


def replay(args: argparse.Namespace) -> None:
    payload, all_frames = load_episode(args.episode_path)
    frames = select_frames(all_frames, args.start_index, args.max_frames)
    frequency = infer_frequency(payload, args.frequency)
    arm_dim, state_arm_dim, left_ee_dim, right_ee_dim, has_action_endpose = validate_frames(
        frames,
        require_action_arm_qpos=args.arm_action_source == "qpos",
        require_action_endpose=args.arm_action_source == "endpose",
    )

    logger.info(
        "loaded %d selected frames from %s at %.3f Hz",
        len(frames),
        resolve_data_path(args.episode_path),
        frequency,
    )
    logger.info(
        "dimensions: action_arm=%s state_arm=%d left_ee=%d right_ee=%d action_endpose=%s",
        "missing" if arm_dim is None else arm_dim,
        state_arm_dim,
        left_ee_dim,
        right_ee_dim,
        "yes" if has_action_endpose else "no",
    )

    ee_name = selected_ee_name(args, left_ee_dim, right_ee_dim)
    use_ee = bool(ee_name)
    robot_arm_dof = ARM_DOF[args.arm]
    robot_ee_dof = 6 if use_ee else 0

    if args.arm_action_source == "qpos" and arm_dim != robot_arm_dof:
        raise ValueError(
            f"raw action arm dim {arm_dim} does not match robot arm_dof {robot_arm_dof}"
        )
    if args.arm_action_source == "endpose" and not has_action_endpose:
        raise ValueError(
            "raw episode does not contain actions.left_arm_ee_pose.pose and "
            "actions.right_arm_ee_pose.pose"
        )
    if state_arm_dim != robot_arm_dof:
        raise ValueError(
            "raw initial state arm dim "
            f"{state_arm_dim} does not match robot arm_dof {robot_arm_dof}"
        )

    if use_ee and (left_ee_dim != robot_ee_dof or right_ee_dim != robot_ee_dof):
        raise ValueError(
            "raw EE dims "
            f"left={left_ee_dim}, right={right_ee_dim} do not match "
            f"robot ee_dof {robot_ee_dof}"
        )

    initial_target_q, initial_full_body_q, initial_pose_source = load_initial_pose_from_config(
        args.initial_pose_config,
        args.initial_pose_task,
        args.arm,
    )
    if initial_target_q.size != robot_arm_dof:
        raise ValueError(
            "initial pose arm dim "
            f"{initial_target_q.size} does not match robot arm_dof {robot_arm_dof}"
        )
    initial_full_body_dim = 0 if initial_full_body_q is None else initial_full_body_q.size

    control_full_body_init = False
    if not args.skip_init and not args.skip_full_body_init:
        if args.arm != "G1_29":
            logger.info("full-body init is only implemented for G1_29; using arm-only init")
        elif args.motion:
            logger.info("full-body init is only supported in debug mode; using arm-only init")
        elif initial_full_body_q is None:
            logger.info("initial pose config has no all_joint_q; using arm-only init")
        elif initial_full_body_q.size != FULL_BODY_DOF["G1_29"]:
            raise ValueError(
                "initial pose all_joint_q dim "
                f"{initial_full_body_q.size} does not match G1_29 full-body dim {FULL_BODY_DOF['G1_29']}"
            )
        else:
            control_full_body_init = True

    logger.info(
        "initialization: task=%s source=%s arm_dim=%d body_dim=%d full_body_init=%s",
        args.initial_pose_task,
        initial_pose_source,
        initial_target_q.size,
        initial_full_body_dim,
        "yes" if control_full_body_init else "no",
    )

    if args.dry_run:
        return

    wait_for_start(args.yes)

    add_repo_root_to_path()
    initialize_dds(sim=args.sim, network_interface=args.network_interface)
    if not args.motion and not args.sim and not args.skip_debug_mode:
        enter_debug_mode(require_success=control_full_body_init)
    elif control_full_body_init and not args.sim and args.skip_debug_mode:
        logger.warning(
            "full-body init is enabled while --skip-debug-mode is set; make sure the robot is already in debug mode."
        )

    arm_ctrl = create_arm_controller(
        args.arm,
        motion=args.motion,
        sim=args.sim,
        initial_target_q=initial_target_q,
        initial_full_body_q=initial_full_body_q if control_full_body_init else None,
        control_full_body=control_full_body_init,
        body_velocity_limit=args.full_body_init_velocity_limit,
        body_ramp_duration=args.full_body_init_ramp_s,
    )
    hand_ctrl = BraincoDirectController() if use_ee else None
    arm_ik = create_arm_ik(args.arm) if args.arm_action_source == "endpose" else None
    solve_tau = (
        create_tau_solver(args.arm, args.tau_source, robot_arm_dof)
        if args.arm_action_source == "qpos" or not args.skip_init
        else None
    )

    if not args.skip_init:
        init_arm_pose = initial_target_q
        if control_full_body_init:
            logger.info(
                "initializing robot to configured full-body state over %.2fs",
                args.full_body_init_ramp_s,
            )
        else:
            logger.info("initializing robot to configured arm state")
        tau = solve_tau(init_arm_pose)
        arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
        if control_full_body_init:
            time.sleep(max(0.0, args.full_body_init_ramp_s))
        time.sleep(max(0.0, args.init_hold_s))

    period = 1.0 / frequency
    logger.info("starting raw replay loop with arm_action_source=%s", args.arm_action_source)
    for frame_offset, frame in enumerate(frames):
        loop_start_time = time.perf_counter()
        left_action_pose = None
        right_action_pose = None
        if args.arm_action_source == "endpose":
            left_action_pose, right_action_pose = arm_endpose(frame, "actions")
            current_q = arm_ctrl.get_current_dual_arm_q()
            current_dq = arm_ctrl.get_current_dual_arm_dq()
            action_q, tau = arm_ik.solve_ik(
                left_action_pose,
                right_action_pose,
                current_q,
                current_dq,
            )
        else:
            action_q = arm_qpos(frame, "actions")
            tau = solve_tau(action_q)
        arm_ctrl.ctrl_dual_arm(action_q, tau)

        if use_ee:
            left_ee_action, right_ee_action = ee_qpos(frame, "actions")
            hand_ctrl.ctrl_dual_hand(left_ee_action, right_ee_action)

        if args.log_every > 0 and frame_offset % args.log_every == 0:
            if args.arm_action_source == "endpose":
                logger.info(
                    "frame=%s left_endpose_xyz=%s right_endpose_xyz=%s ik_q=%s tau=%s",
                    frame.get("idx", "?"),
                    left_action_pose[:3, 3],
                    right_action_pose[:3, 3],
                    action_q,
                    tau,
                )
            else:
                logger.info(
                    "frame=%s arm_action=%s tau=%s",
                    frame.get("idx", "?"),
                    action_q,
                    tau,
                )

        elapsed = time.perf_counter() - loop_start_time
        time.sleep(max(0.0, period - elapsed))

    logger.info("raw replay finished")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        replay(args)
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
