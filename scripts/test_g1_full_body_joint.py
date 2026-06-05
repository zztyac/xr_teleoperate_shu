#!/usr/bin/env python3
"""Write a G1_29 full-body joint target through rt/lowcmd.

Example:
    python3 scripts/test_g1_full_body_joint.py --yes
    python3 scripts/test_g1_full_body_joint.py --duration 8 --hold 10 --yes

The script reads teleop/initial_target_poses.json by default and writes the
G1_29 all_joint_q target for valid joints 0..28. Joints 29..34 are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

from teleop.robot_control.robot_arm import (
    G1_29_JointIndex,
    G1_29_LowLevel_Kd,
    G1_29_LowLevel_Kp,
    G1_29_Valid_Motors,
    G1_Mode_PR,
    kTopicLowCommand_Debug,
    kTopicLowState,
)
from teleop.utils.motion_switcher import MotionSwitcher


DEFAULT_CONFIG = REPO_ROOT / "teleop" / "initial_target_poses.json"
DEFAULT_TASK = "vehicle_physical_button_press_ccw"
ARM_NAME = "G1_29"


def load_all_joint_q(config_path: Path, task_name: str) -> np.ndarray:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        entry = config[task_name]
    except KeyError as exc:
        raise KeyError(f"Task '{task_name}' not found in {config_path}") from exc

    if isinstance(entry, dict) and ARM_NAME in entry:
        entry = entry[ARM_NAME]

    if isinstance(entry, dict) and "all_joint_q" in entry:
        raw_q = entry["all_joint_q"]
    elif isinstance(entry, list):
        raw_q = entry
    else:
        raise ValueError(f"Task '{task_name}' must contain {ARM_NAME}.all_joint_q")

    q = np.asarray(raw_q, dtype=float).reshape(-1)
    expected = len(G1_29_JointIndex)
    if q.shape[0] != expected:
        raise ValueError(f"all_joint_q must have {expected} values, got {q.shape[0]}")
    return q


def wait_lowstate(subscriber: ChannelSubscriber, timeout: float) -> hg_LowState:
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = subscriber.Read()
        if msg is not None:
            return msg
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {kTopicLowState} after {timeout:.1f}s")


def lowstate_q(lowstate: hg_LowState) -> np.ndarray:
    return np.array([float(lowstate.motor_state[i].q) for i in range(G1_29_Valid_Motors)], dtype=float)


def init_lowcmd(start_q: np.ndarray, kp_scale: float, kd_scale: float):
    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = G1_Mode_PR
    for i in range(G1_29_Valid_Motors):
        msg.motor_cmd[i].mode = 1
        msg.motor_cmd[i].q = float(start_q[i])
        msg.motor_cmd[i].dq = 0.0
        msg.motor_cmd[i].tau = 0.0
        msg.motor_cmd[i].kp = float(G1_29_LowLevel_Kp[i] * kp_scale)
        msg.motor_cmd[i].kd = float(G1_29_LowLevel_Kd[i] * kd_scale)
    return msg


def print_preview(start_q: np.ndarray, target_q: np.ndarray, control_indices: list[int]) -> None:
    body_indices = [i for i in control_indices if i < 15]
    arm_indices = [i for i in control_indices if 15 <= i < G1_29_Valid_Motors]
    print("Target source: teleop/initial_target_poses.json")
    print(f"Control valid joints: {control_indices}")
    print("Current body q[0:15]:", np.array2string(start_q[:15], precision=6, separator=", "))
    print("Target  body q[0:15]:", np.array2string(target_q[:15], precision=6, separator=", "))
    if arm_indices:
        print("Current arm  q[15:29]:", np.array2string(start_q[15:29], precision=6, separator=", "))
        print("Target  arm  q[15:29]:", np.array2string(target_q[15:29], precision=6, separator=", "))
    print(f"Body joints controlled: {body_indices}")
    print(f"Arm joints controlled: {arm_indices}")


def confirm_or_exit(args) -> None:
    if args.yes or args.dry_run:
        return
    print()
    print("WARNING: This will publish G1_29 low-level joint commands to rt/lowcmd.")
    print("Keep the robot supported and clear of obstacles.")
    answer = input("Type 'yes' to continue: ").strip().lower()
    if answer != "yes":
        raise SystemExit("Aborted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test G1_29 full-body joint command through rt/lowcmd.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to initial_target_poses.json.")
    parser.add_argument("--task-name", default=DEFAULT_TASK, help="Task key to read from the config.")
    parser.add_argument("--duration", type=float, default=6.0, help="Seconds to ramp from current q to target q.")
    parser.add_argument("--hold", type=float, default=5.0, help="Seconds to keep publishing the target after ramp.")
    parser.add_argument("--rate", type=float, default=250.0, help="Command publish rate in Hz.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for one lowstate message.")
    parser.add_argument("--network-interface", default=None, help="DDS network interface, e.g. eth0.")
    parser.add_argument("--sim", action="store_true", help="Use DDS domain 1. Default is real robot domain 0.")
    parser.add_argument("--skip-debug-mode", action="store_true", help="Do not call MotionSwitcher.Enter_Debug_Mode().")
    parser.add_argument("--body-only", action="store_true", help="Only control leg and waist joints 0..14.")
    parser.add_argument("--kp-scale", type=float, default=1.0, help="Scale SDK G1 low-level Kp gains.")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="Scale SDK G1 low-level Kd gains.")
    parser.add_argument("--dry-run", action="store_true", help="Print target/current q and exit without publishing.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than 0.")
    if args.hold < 0:
        parser.error("--hold must be greater than or equal to 0.")
    if args.rate <= 0:
        parser.error("--rate must be greater than 0.")
    if args.kp_scale <= 0 or args.kd_scale <= 0:
        parser.error("--kp-scale and --kd-scale must be greater than 0.")

    full_target_q = load_all_joint_q(args.config, args.task_name)
    target_q = full_target_q[:G1_29_Valid_Motors].copy()
    control_indices = list(range(15)) if args.body_only else list(range(G1_29_Valid_Motors))

    if args.dry_run:
        print(f"Task: {args.task_name}")
        print(f"Control valid joints: {control_indices}")
        print("Target body q[0:15]:", np.array2string(target_q[:15], precision=6, separator=", "))
        print("Target arm  q[15:29]:", np.array2string(target_q[15:29], precision=6, separator=", "))
        return

    domain_id = 1 if args.sim else 0
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)

    if not args.sim and not args.skip_debug_mode and not args.dry_run:
        status, result = MotionSwitcher().Enter_Debug_Mode()
        mode_name = result.get("name") if isinstance(result, dict) else None
        if status != 0 or result is None or mode_name:
            raise RuntimeError(f"Failed to enter debug mode: status={status}, result={result}")
        print("Entered debug mode.")

    lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_subscriber.Init()
    start_state = wait_lowstate(lowstate_subscriber, args.timeout)
    start_q = lowstate_q(start_state)

    print_preview(start_q, target_q, control_indices)
    confirm_or_exit(args)

    lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_publisher.Init()
    low_cmd = init_lowcmd(start_q, args.kp_scale, args.kd_scale)
    crc = CRC()

    dt = 1.0 / args.rate
    ramp_steps = max(1, int(args.duration * args.rate))
    hold_steps = int(args.hold * args.rate)
    total_steps = ramp_steps + hold_steps
    last_report = 0.0
    print(f"Publishing {total_steps} steps at {args.rate:.1f} Hz...")

    try:
        for step in range(total_steps):
            now = time.time()
            lowstate = lowstate_subscriber.Read()
            if lowstate is not None:
                low_cmd.mode_machine = lowstate.mode_machine

            ratio = min(1.0, (step + 1) / ramp_steps)
            commanded_q = start_q + (target_q - start_q) * ratio

            for i in control_indices:
                low_cmd.motor_cmd[i].q = float(commanded_q[i])
                low_cmd.motor_cmd[i].dq = 0.0
                low_cmd.motor_cmd[i].tau = 0.0

            low_cmd.crc = crc.Crc(low_cmd)
            lowcmd_publisher.Write(low_cmd)

            if now - last_report >= 1.0:
                current_state = lowstate or wait_lowstate(lowstate_subscriber, 0.2)
                current_q = lowstate_q(current_state)
                err = np.abs(target_q[control_indices] - current_q[control_indices])
                print(
                    f"t={step * dt:6.2f}s ratio={ratio:.3f} "
                    f"max_err={float(np.max(err)):.4f} "
                    f"body_q={np.array2string(current_q[:15], precision=4, separator=', ')}"
                )
                last_report = now

            elapsed = time.time() - now
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping publisher.")
    finally:
        final_state = wait_lowstate(lowstate_subscriber, args.timeout)
        final_q = lowstate_q(final_state)
        print("Final body q[0:15]:", np.array2string(final_q[:15], precision=6, separator=", "))
        print("Final arm  q[15:29]:", np.array2string(final_q[15:29], precision=6, separator=", "))


if __name__ == "__main__":
    main()
