"""Replay a recorded episode on the real robot.

Default target data:
  /mnt/data/zty/json_data/vehicle_physical_button_press/episode_0003

Example:
  python teleop/replay_episode_real.py --yes
  python teleop/replay_episode_real.py --dry-run

This script replays the recorded joint-space data directly. It does not start
XR, cameras, IK, or data recording.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import logging_mp

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

from teleop.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_23_ArmController,
    H1_2_ArmController,
    H1_ArmController,
    H2_ArmController,
)
from teleop.robot_control.robot_hand_brainco import (
    Brainco_Left_Hand_JointIndex,
    Brainco_Right_Hand_JointIndex,
    kTopicbraincoLeftCommand,
    kTopicbraincoRightCommand,
)
from teleop.utils.motion_switcher import MotionSwitcher


DEFAULT_EPISODE_DIR = "/mnt/data/zty/json_data/vehicle_physical_button_press/episode_0003"

ARM_CONTROLLER_BY_NAME = {
    "G1_29": G1_29_ArmController,
    "G1_23": G1_23_ArmController,
    "H1_2": H1_2_ArmController,
    "H1": H1_ArmController,
    "H2": H2_ArmController,
}

ARM_DOF_BY_NAME = {
    "G1_29": 14,
    "G1_23": 10,
    "H1_2": 14,
    "H1": 8,
    "H2": 14,
}


class BraincoReplayController:
    """Minimal direct BrainCo hand command publisher for recorded qpos."""

    def __init__(self):
        self.left_publisher = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
        self.left_publisher.Init()
        self.right_publisher = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
        self.right_publisher.Init()

        self.left_msg = MotorCmds_()
        self.left_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Left_Hand_JointIndex))
        ]
        self.right_msg = MotorCmds_()
        self.right_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Right_Hand_JointIndex))
        ]
        for cmd in self.left_msg.cmds + self.right_msg.cmds:
            cmd.q = 0.0
            cmd.dq = 1.0

    def publish(self, left_q, right_q):
        left_q = np.clip(np.asarray(left_q, dtype=float), 0.0, 1.0)
        right_q = np.clip(np.asarray(right_q, dtype=float), 0.0, 1.0)
        if left_q.shape[0] != len(Brainco_Left_Hand_JointIndex):
            raise ValueError(f"left BrainCo action must have 6 values, got {left_q.shape[0]}.")
        if right_q.shape[0] != len(Brainco_Right_Hand_JointIndex):
            raise ValueError(f"right BrainCo action must have 6 values, got {right_q.shape[0]}.")

        for idx, motor_id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_msg.cmds[motor_id].q = float(left_q[idx])
        for idx, motor_id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_msg.cmds[motor_id].q = float(right_q[idx])

        self.left_publisher.Write(self.left_msg)
        self.right_publisher.Write(self.right_msg)

    def open(self, hold_seconds=0.5, frequency=30.0):
        zero = np.zeros(6)
        for _ in range(max(1, int(hold_seconds * frequency))):
            self.publish(zero, zero)
            time.sleep(1.0 / frequency)


def load_episode(episode_dir):
    data_path = os.path.join(episode_dir, "data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        episode = json.load(f)

    frames = episode.get("data", [])
    if not frames:
        raise ValueError(f"No frames found in {data_path}.")
    return episode, frames


def frame_qpos(frame, source, key):
    return frame.get(source, {}).get(key, {}).get("qpos", [])


def frame_action(frame, key):
    return frame_qpos(frame, "actions", key)


def dual_arm_qpos(frame, source):
    left = frame_qpos(frame, source, "left_arm")
    right = frame_qpos(frame, source, "right_arm")
    return np.asarray(left + right, dtype=float)


def validate_frames(frames, arm_dof, hand_enabled, arm_source):
    for pos, frame in enumerate(frames):
        q = dual_arm_qpos(frame, arm_source)
        if q.shape[0] != arm_dof:
            raise ValueError(
                f"Frame {pos} arm {arm_source} qpos has {q.shape[0]} values, expected {arm_dof}."
            )
        if hand_enabled:
            left_ee = frame_action(frame, "left_ee")
            right_ee = frame_action(frame, "right_ee")
            if len(left_ee) != 6 or len(right_ee) != 6:
                raise ValueError(
                    f"Frame {pos} BrainCo action must be left/right 6 values, "
                    f"got {len(left_ee)} and {len(right_ee)}."
                )


def slice_frames(frames, start_index, end_index):
    if start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if end_index is None:
        end_index = len(frames)
    if end_index <= start_index:
        raise ValueError("--end-index must be greater than --start-index.")
    return frames[start_index:end_index]


def ramp_arm_to(arm_ctrl, target_q, seconds, frequency):
    if seconds <= 0:
        arm_ctrl.ctrl_dual_arm(target_q, np.zeros_like(target_q))
        return

    current_q = arm_ctrl.get_current_dual_arm_q()
    steps = max(1, int(seconds * frequency))
    for step in range(1, steps + 1):
        alpha = step / steps
        q = current_q * (1.0 - alpha) + target_q * alpha
        arm_ctrl.ctrl_dual_arm(q, np.zeros_like(q))
        time.sleep(1.0 / frequency)


def wait_arm_near(arm_ctrl, target_q, tolerance, timeout):
    if timeout <= 0:
        return

    start_time = time.time()
    while True:
        current_q = arm_ctrl.get_current_dual_arm_q()
        max_error = float(np.max(np.abs(current_q - target_q)))
        if max_error <= tolerance:
            logger_mp.info(f"Arm reached initial replay pose, max error {max_error:.4f} rad.")
            return
        if time.time() - start_time >= timeout:
            logger_mp.warning(
                f"Arm did not fully settle before replay, max error {max_error:.4f} rad "
                f"> tolerance {tolerance:.4f} rad."
            )
            return
        arm_ctrl.ctrl_dual_arm(target_q, np.zeros_like(target_q))
        time.sleep(0.02)


def create_arm_controller(arm_name, motion_mode, simulation_mode, initial_target_q):
    arm_cls = ARM_CONTROLLER_BY_NAME[arm_name]
    if arm_name == "H1":
        return arm_cls(simulation_mode=simulation_mode, initial_target_q=initial_target_q)
    return arm_cls(
        motion_mode=motion_mode,
        simulation_mode=simulation_mode,
        initial_target_q=initial_target_q,
    )


def replay(frames, arm_ctrl, hand_ctrl, frequency, speed, arm_source):
    dt = 1.0 / (frequency * speed)
    tau = np.zeros_like(dual_arm_qpos(frames[0], arm_source))

    for replay_idx, frame in enumerate(frames):
        start_time = time.time()
        arm_q = dual_arm_qpos(frame, arm_source)
        arm_ctrl.ctrl_dual_arm(arm_q, tau)

        if hand_ctrl is not None:
            hand_ctrl.publish(frame_action(frame, "left_ee"), frame_action(frame, "right_ee"))

        if replay_idx % max(1, int(frequency)) == 0:
            logger_mp.info(f"Replayed {replay_idx + 1}/{len(frames)} frames.")

        elapsed = time.time() - start_time
        time.sleep(max(0.0, dt - elapsed))


def parse_args():
    parser = argparse.ArgumentParser(description="Replay recorded real-robot joint actions.")
    parser.add_argument("--episode-dir", type=str, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--arm", choices=sorted(ARM_CONTROLLER_BY_NAME), default="G1_29")
    parser.add_argument("--ee", choices=["brainco", "none"], default="brainco")
    parser.add_argument(
        "--arm-source",
        choices=["states", "actions"],
        default="states",
        help="Use recorded measured arm states or recorded control actions.",
    )
    parser.add_argument("--frequency", type=float, default=None, help="Override replay frequency.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive end frame index.")
    parser.add_argument("--ramp-seconds", type=float, default=3.0)
    parser.add_argument("--arm-velocity-limit", type=float, default=8.0)
    parser.add_argument("--settle-tolerance", type=float, default=0.04)
    parser.add_argument("--settle-timeout", type=float, default=3.0)
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--motion", action="store_true", help="Use motion-mode arm topic instead of debug topic.")
    parser.add_argument("--sim", action="store_true", help="Use DDS domain 1 for simulator checks.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate and print episode summary.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive safety confirmation.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be > 0.")

    episode, all_frames = load_episode(args.episode_dir)
    frames = slice_frames(all_frames, args.start_index, args.end_index)
    frequency = args.frequency or episode.get("info", {}).get("image", {}).get("fps") or 30.0
    arm_dof = ARM_DOF_BY_NAME[args.arm]
    hand_enabled = args.ee == "brainco"
    validate_frames(frames, arm_dof, hand_enabled, args.arm_source)

    text = episode.get("text", {})
    logger_mp.info(f"Episode: {args.episode_dir}")
    logger_mp.info(f"Goal: {text.get('goal', '')}")
    logger_mp.info(
        f"Frames: {len(frames)} / {len(all_frames)}, frequency: {frequency} Hz, "
        f"speed: {args.speed}x, arm: {args.arm}, arm_source: {args.arm_source}, ee: {args.ee}"
    )

    if args.dry_run:
        first_q = dual_arm_qpos(frames[0], args.arm_source)
        last_q = dual_arm_qpos(frames[-1], args.arm_source)
        action0 = dual_arm_qpos(frames[0], "actions")
        state0 = dual_arm_qpos(frames[0], "states")
        logger_mp.info(f"First arm {args.arm_source}: {np.round(first_q, 4).tolist()}")
        logger_mp.info(f"Last arm {args.arm_source}:  {np.round(last_q, 4).tolist()}")
        logger_mp.info(
            f"Frame-0 action/state max diff: {float(np.max(np.abs(action0 - state0))):.4f} rad"
        )
        return

    if not args.yes:
        answer = input("This will move the real robot. Type 'replay' to start: ").strip()
        if answer != "replay":
            logger_mp.info("Cancelled.")
            return

    domain_id = 1 if args.sim else 0
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)

    motion_switcher = None
    if not args.motion and not args.sim:
        motion_switcher = MotionSwitcher()
        status, result = motion_switcher.Enter_Debug_Mode()
        logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'} {result}")

    first_arm_q = dual_arm_qpos(frames[0], args.arm_source)
    arm_ctrl = create_arm_controller(args.arm, args.motion, args.sim, first_arm_q)
    arm_ctrl.arm_velocity_limit = args.arm_velocity_limit
    hand_ctrl = BraincoReplayController() if hand_enabled else None

    try:
        logger_mp.info(f"Ramping arm to first recorded action for {args.ramp_seconds:.1f}s.")
        ramp_arm_to(arm_ctrl, first_arm_q, args.ramp_seconds, frequency)
        wait_arm_near(arm_ctrl, first_arm_q, args.settle_tolerance, args.settle_timeout)
        if hand_ctrl is not None:
            hand_ctrl.publish(frame_action(frames[0], "left_ee"), frame_action(frames[0], "right_ee"))
            time.sleep(0.3)

        logger_mp.info("Replay started.")
        replay(frames, arm_ctrl, hand_ctrl, frequency, args.speed, args.arm_source)
        logger_mp.info("Replay finished.")
    except KeyboardInterrupt:
        logger_mp.info("Interrupted by user.")
    finally:
        try:
            if hand_ctrl is not None:
                hand_ctrl.open(frequency=frequency)
        except Exception as e:
            logger_mp.error(f"Failed to open BrainCo hand: {e}")
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to move arm back to initial replay pose: {e}")
        if motion_switcher is not None:
            logger_mp.info("Leaving robot mode unchanged after replay.")


if __name__ == "__main__":
    main()
