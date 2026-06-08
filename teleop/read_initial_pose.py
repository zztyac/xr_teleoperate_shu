#!/usr/bin/env python3
"""Read the robot's current joint pose without publishing commands.

python teleop/read_initial_pose.py --arm G1_29 --task-name vehicle_physical_button_press_ccw

数据来源说明：
  这个脚本不读取 initial_target_poses.json，也不会向机器人下发控制命令。
  它只订阅 DDS 里的 rt/lowstate 实时状态消息，从 motor_state 中取当前关节 q 值，
  然后把结果直接写入 teleop/initial_target_poses.json，作为后续遥操作初始姿态。

"""

import argparse
import json
import os
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
    # 每种机器人型号对应不同的 LowState 消息类型和关节枚举。
    # all_joints 用于完整机身初始姿态，arm_joints 只取双臂关节初始姿态。
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
INITIAL_TARGET_POSES_PATH = os.path.join(os.path.dirname(__file__), "initial_target_poses.json")


def read_lowstate(subscriber: ChannelSubscriber, timeout: float):
    # DDS 订阅是实时消息流；这里循环等待直到读到一帧 rt/lowstate。
    # 读到的 msg 就是机器人当前状态快照，脚本只读取它，不发布 lowcmd。
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = subscriber.Read()
        if msg is not None:
            return msg
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {kTopicLowState} after {timeout:.1f}s")


def values_as_array(lowstate, joints: Iterable, attr: str, precision: int | None) -> list[float]:
    # joints 是具体型号的关节枚举；joint.value 是 motor_state 里的索引。
    # attr 当前传入 "q"，所以这里读取的是每个关节当前位置/角度。
    values = [float(getattr(lowstate.motor_state[joint.value], attr)) for joint in joints]
    if precision is None:
        return values
    return [round(value, precision) for value in values]


def load_initial_target_poses(path: str) -> dict:
    # 先读取已有 JSON，保留其他 task-name / arm 型号的数据，只覆盖本次指定的条目。
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object at top level")
    return data


def write_initial_target_pose(path: str, task_name: str, arm: str, pose: dict) -> None:
    data = load_initial_target_poses(path)
    task_data = data.setdefault(task_name, {})
    if not isinstance(task_data, dict):
        raise ValueError(f"{path}[{task_name!r}] must contain a JSON object")
    task_data[arm] = pose
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read current robot joint state from rt/lowstate only. No lowcmd is published."
    )
    parser.add_argument("--arm", choices=ARM_CONFIGS.keys(), default="G1_29", help="Robot arm model.")
    parser.add_argument("--sim", action="store_true", help="Use DDS domain 1 for simulation. Default is domain 0.")
    parser.add_argument("--network-interface", default=None, help="DDS network interface, e.g. eth0.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for one lowstate message.")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME, help="Task key to write in initial_target_poses.json.")
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal places for q values. Use a negative value to keep raw floats.",
    )
    args = parser.parse_args()

    cfg = ARM_CONFIGS[args.arm]
    domain_id = 1 if args.sim else 0
    # 初始化 Unitree DDS 通道：实机默认 domain 0，仿真用 --sim 切到 domain 1。
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)

    # 订阅 rt/lowstate。消息类型由机器人型号决定，避免用错 go/hg LowState 结构。
    subscriber = ChannelSubscriber(kTopicLowState, cfg["lowstate_type"])
    subscriber.Init()
    lowstate = read_lowstate(subscriber, args.timeout)

    # 按 robot_arm.py 中定义的枚举顺序取值，保证输出数组顺序和控制代码读取顺序一致。
    all_joints = list(cfg["all_joints"])
    arm_joints = list(cfg["arm_joints"])
    precision = None if args.precision < 0 else args.precision
    # 写入结构对齐 teleop/initial_target_poses.json：
    # task-name -> arm 型号 -> all_joint_q / dual_arm_q。
    pose = {
        "all_joint_q": values_as_array(lowstate, all_joints, "q", precision),
        "dual_arm_q": values_as_array(lowstate, arm_joints, "q", precision),
    }

    write_initial_target_pose(INITIAL_TARGET_POSES_PATH, args.task_name, args.arm, pose)
    print(f"写入成功: {INITIAL_TARGET_POSES_PATH} [{args.task_name} -> {args.arm}]")


if __name__ == "__main__":
    main()
