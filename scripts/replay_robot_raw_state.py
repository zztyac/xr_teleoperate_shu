#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_EPISODE_PATH = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press/episode_0024"
)
ARM_DOF = {"G1_29": 14, "G1_23": 10}


logger = logging.getLogger("replay_robot_raw_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay raw Unitree JSON states.")
    parser.add_argument(
        "--arm",
        choices=["G1_29", "G1_23"],
        default="G1_29",
        help="Arm controller type.",
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
        help="Do not replay end-effector states.",
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
        "--frequency",
        type=float,
        default=None,
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
        "--skip-init",
        action="store_true",
        help="Do not move to the first selected state before replay.",
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
        help="Log one state target every N frames. Use 1 to log every frame.",
    )
    return parser.parse_args()


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


def arm_state_qpos(frame: dict) -> np.ndarray:
    return np.concatenate(
        (qpos(frame, "states", "left_arm"), qpos(frame, "states", "right_arm"))
    )


def ee_state_qpos(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    return qpos(frame, "states", "left_ee"), qpos(frame, "states", "right_ee")


def infer_frequency(payload: dict, override: float | None) -> float:
    if override is not None:
        frequency = override
    else:
        frequency = payload.get("info", {}).get("image", {}).get("fps", 15)

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


def validate_frames(frames: list[dict]) -> tuple[int, int, int]:
    first = frames[0]
    arm_dim = arm_state_qpos(first).size
    left_ee_dim, right_ee_dim = (arr.size for arr in ee_state_qpos(first))

    for frame in frames:
        idx = frame.get("idx", "?")
        if arm_state_qpos(frame).size != arm_dim:
            raise ValueError(f"frame {idx} state arm dimension changed")
        left_ee, right_ee = ee_state_qpos(frame)
        if left_ee.size != left_ee_dim or right_ee.size != right_ee_dim:
            raise ValueError(f"frame {idx} EE state dimension changed")

    return arm_dim, left_ee_dim, right_ee_dim


def add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def initialize_dds(sim: bool, network_interface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    domain_id = 1 if sim else 0
    if network_interface:
        ChannelFactoryInitialize(domain_id, networkInterface=network_interface)
    else:
        ChannelFactoryInitialize(domain_id)


def enter_debug_mode() -> None:
    from teleop.utils.motion_switcher import MotionSwitcher

    status, result = MotionSwitcher().Enter_Debug_Mode()
    logger.info("Enter debug mode: %s %s", "Success" if status == 0 else "Failed", result)


def create_arm_controller(arm_name: str, motion: bool, sim: bool):
    from teleop.robot_control.robot_arm import G1_23_ArmController, G1_29_ArmController

    controllers = {
        "G1_29": G1_29_ArmController,
        "G1_23": G1_23_ArmController,
    }
    return controllers[arm_name](motion_mode=motion, simulation_mode=sim)


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
        "Please enter the start signal (enter 's' to start state replay): "
    )
    if user_input.lower() != "s":
        raise SystemExit("Replay cancelled.")


def replay(args: argparse.Namespace) -> None:
    payload, all_frames = load_episode(args.episode_path)
    frames = select_frames(all_frames, args.start_index, args.max_frames)
    frequency = infer_frequency(payload, args.frequency)
    arm_dim, left_ee_dim, right_ee_dim = validate_frames(frames)

    logger.info(
        "loaded %d selected frames from %s at %.3f Hz",
        len(frames),
        resolve_data_path(args.episode_path),
        frequency,
    )
    logger.info(
        "state dimensions: arm=%d left_ee=%d right_ee=%d",
        arm_dim,
        left_ee_dim,
        right_ee_dim,
    )

    if args.dry_run:
        return

    ee_name = selected_ee_name(args, left_ee_dim, right_ee_dim)
    use_ee = bool(ee_name)
    robot_arm_dof = ARM_DOF[args.arm]
    robot_ee_dof = 6 if use_ee else 0

    if arm_dim != robot_arm_dof:
        raise ValueError(
            f"raw state arm dim {arm_dim} does not match robot arm_dof {robot_arm_dof}"
        )
    if use_ee and (left_ee_dim != robot_ee_dof or right_ee_dim != robot_ee_dof):
        raise ValueError(
            "raw EE state dims "
            f"left={left_ee_dim}, right={right_ee_dim} do not match "
            f"robot ee_dof {robot_ee_dof}"
        )

    wait_for_start(args.yes)

    add_repo_root_to_path()
    initialize_dds(sim=args.sim, network_interface=args.network_interface)
    if not args.motion and not args.sim and not args.skip_debug_mode:
        enter_debug_mode()

    arm_ctrl = create_arm_controller(args.arm, motion=args.motion, sim=args.sim)
    hand_ctrl = BraincoDirectController() if use_ee else None
    tau = np.zeros(robot_arm_dof, dtype=np.float64)

    if not args.skip_init:
        init_arm_pose = arm_state_qpos(frames[0])
        logger.info("initializing robot to first selected recorded state")
        arm_ctrl.ctrl_dual_arm(init_arm_pose, tau)
        time.sleep(max(0.0, args.init_hold_s))

    period = 1.0 / frequency
    logger.info("starting raw state replay loop")
    for frame_offset, frame in enumerate(frames):
        loop_start_time = time.perf_counter()
        arm_state = arm_state_qpos(frame)
        arm_ctrl.ctrl_dual_arm(arm_state, tau)

        if use_ee:
            left_ee_state, right_ee_state = ee_state_qpos(frame)
            hand_ctrl.ctrl_dual_hand(left_ee_state, right_ee_state)

        if args.log_every > 0 and frame_offset % args.log_every == 0:
            logger.info("frame=%s arm_state=%s", frame.get("idx", "?"), arm_state)

        elapsed = time.perf_counter() - loop_start_time
        time.sleep(max(0.0, period - elapsed))

    logger.info("raw state replay finished")


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
