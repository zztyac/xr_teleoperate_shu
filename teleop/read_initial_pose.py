#!/usr/bin/env python3
"""Read the robot's current joint pose without publishing commands.

python teleop/read_initial_pose.py --arm G1_29 --task-name vehicle_physical_button_press_ccw

"""

import argparse
import json
import time
from typing import Iterable

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as go_LowState
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

from robot_control.robot_arm import (
    G1_23_JointArmIndex,
    G1_23_JointIndex,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    H1_2_JointArmIndex,
    H1_2_JointIndex,
    H1_JointArmIndex,
    H1_JointIndex,
    H2_JointArmIndex,
    H2_JointIndex,
    kTopicLowState,
)


ARM_CONFIGS = {
    "G1_29": {
        "lowstate_type": hg_LowState,
        "all_joints": G1_29_JointIndex,
        "arm_joints": G1_29_JointArmIndex,
    },
    "G1_23": {
        "lowstate_type": hg_LowState,
        "all_joints": G1_23_JointIndex,
        "arm_joints": G1_23_JointArmIndex,
    },
    "H1_2": {
        "lowstate_type": hg_LowState,
        "all_joints": H1_2_JointIndex,
        "arm_joints": H1_2_JointArmIndex,
    },
    "H1": {
        "lowstate_type": go_LowState,
        "all_joints": H1_JointIndex,
        "arm_joints": H1_JointArmIndex,
    },
    "H2": {
        "lowstate_type": hg_LowState,
        "all_joints": H2_JointIndex,
        "arm_joints": H2_JointArmIndex,
    },
}


DEFAULT_TASK_NAME = "vehicle_physical_button_press"
DEFAULT_PRECISION = 6


def read_lowstate(subscriber: ChannelSubscriber, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = subscriber.Read()
        if msg is not None:
            return msg
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {kTopicLowState} after {timeout:.1f}s")


def values_as_array(lowstate, joints: Iterable, attr: str, precision: int | None) -> list[float]:
    values = [float(getattr(lowstate.motor_state[joint.value], attr)) for joint in joints]
    if precision is None:
        return values
    return [round(value, precision) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read current robot joint state from rt/lowstate only. No lowcmd is published."
    )
    parser.add_argument("--arm", choices=ARM_CONFIGS.keys(), default="G1_29", help="Robot arm model.")
    parser.add_argument("--sim", action="store_true", help="Use DDS domain 1 for simulation. Default is domain 0.")
    parser.add_argument("--network-interface", default=None, help="DDS network interface, e.g. eth0.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for one lowstate message.")
    parser.add_argument("--all", action="store_true", help="Kept for compatibility; all body joints are printed by default.")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME, help="Task key to print for initial_target_poses.json.")
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal places for q values. Use a negative value to keep raw floats.",
    )
    parser.add_argument("--compact", action="store_true", help="Print compact JSON instead of indented JSON.")
    parser.add_argument("--pretty", action="store_true", help="Kept for compatibility; JSON is indented by default.")
    args = parser.parse_args()

    cfg = ARM_CONFIGS[args.arm]
    domain_id = 1 if args.sim else 0
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)

    subscriber = ChannelSubscriber(kTopicLowState, cfg["lowstate_type"])
    subscriber.Init()
    lowstate = read_lowstate(subscriber, args.timeout)

    all_joints = list(cfg["all_joints"])
    arm_joints = list(cfg["arm_joints"])
    precision = None if args.precision < 0 else args.precision
    result = {
        args.task_name: {
            args.arm: {
                "all_joint_q": values_as_array(lowstate, all_joints, "q", precision),
                "dual_arm_q": values_as_array(lowstate, arm_joints, "q", precision),
            }
        }
    }

    indent = None if args.compact else 2
    print(json.dumps(result, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
