#!/usr/bin/env python3
"""Read the robot's current joint pose without publishing commands.

python teleop/read_initial_pose.py --arm G1_29 --pretty

"""

import argparse
import json
import time
from enum import IntEnum
from typing import Iterable

import numpy as np
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


def read_lowstate(subscriber: ChannelSubscriber, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = subscriber.Read()
        if msg is not None:
            return msg
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {kTopicLowState} after {timeout:.1f}s")


def read_joint_values(lowstate, joints: Iterable[IntEnum]) -> dict:
    values = {}
    for joint in joints:
        motor = lowstate.motor_state[joint.value]
        values[joint.name] = {
            "index": int(joint.value),
            "q": float(motor.q),
            "dq": float(motor.dq),
        }
    return values


def values_as_array(lowstate, joints: Iterable[IntEnum], attr: str) -> list[float]:
    return [float(getattr(lowstate.motor_state[joint.value], attr)) for joint in joints]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read current robot joint state from rt/lowstate only. No lowcmd is published."
    )
    parser.add_argument("--arm", choices=ARM_CONFIGS.keys(), default="G1_29", help="Robot arm model.")
    parser.add_argument("--sim", action="store_true", help="Use DDS domain 1 for simulation. Default is domain 0.")
    parser.add_argument("--network-interface", default=None, help="DDS network interface, e.g. eth0.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for one lowstate message.")
    parser.add_argument("--all", action="store_true", help="Also print all named body joints.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    cfg = ARM_CONFIGS[args.arm]
    domain_id = 1 if args.sim else 0
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)

    subscriber = ChannelSubscriber(kTopicLowState, cfg["lowstate_type"])
    subscriber.Init()
    lowstate = read_lowstate(subscriber, args.timeout)

    arm_joints = list(cfg["arm_joints"])
    result = {
        "arm": args.arm,
        "topic": kTopicLowState,
        "domain_id": domain_id,
        "timestamp_unix": time.time(),
        "dual_arm_q": values_as_array(lowstate, arm_joints, "q"),
        "dual_arm_dq": values_as_array(lowstate, arm_joints, "dq"),
        "dual_arm_named": read_joint_values(lowstate, arm_joints),
    }

    if args.all:
        result["all_named"] = read_joint_values(lowstate, cfg["all_joints"])

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))

    if args.pretty:
        q = np.array(result["dual_arm_q"])
        print("\n# Python literal for current dual-arm q:")
        print(np.array2string(q, precision=6, separator=", "))


if __name__ == "__main__":
    main()
