#!/usr/bin/env python3
"""Replay an EpisodeWriter dataset.

Default dataset:
    /mnt/data/zty/json_data/vehicle_physical_button_press/episode_0021/data.json.

Examples:
    python scripts/replay_episode.py --execute
    python scripts/replay_episode.py --camera color_0 --camera color_1 --camera color_2 --camera color_3
    python scripts/replay_episode.py --export-video /tmp/episode_0001.mp4 --no-gui

    python scripts/replay_episode.py --execute --yes --camera color_0 --camera color_1

    python scripts/replay_episode.py --execute --yes --source states --no-gui --fps 30
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_EPISODE_DIR = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press/episode_0001"
)
DEFAULT_TILE_WIDTH = 640
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_TARGET_Q_FILE = REPO_ROOT / "teleop" / "initial_target_poses.json"
ARM_TARGET_DOF = {
    "G1_29": 14,
    "G1_23": 10,
    "H1_2": 14,
    "H1": 8,
    "H2": 14,
}
INITIAL_TARGET_Q_MISSING = object()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay images and states/actions from an EpisodeWriter episode."
    )
    parser.add_argument(
        "episode_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EPISODE_DIR,
        help=f"Episode directory. Default: {DEFAULT_EPISODE_DIR}",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera key to show, for example color_0. Repeat to select multiple cameras.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS. Defaults to info.image.fps from data.json.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 2.0 is twice as fast.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End frame index, inclusive. Defaults to the last frame.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--tile-width",
        type=int,
        default=DEFAULT_TILE_WIDTH,
        help="Width of each camera tile in pixels.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print a joint summary every N frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open an OpenCV window. Useful on headless machines.",
    )
    parser.add_argument(
        "--export-video",
        type=Path,
        default=None,
        help="Optional output video path, for example /tmp/episode_0001.mp4.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send replay commands to the robot instead of only visualizing data.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required together with --execute to acknowledge physical robot motion.",
    )
    parser.add_argument(
        "--arm",
        choices=["none", "G1_29", "G1_23", "H1_2", "H1", "H2"],
        default="G1_29",
        help="Arm controller used by --execute.",
    )
    parser.add_argument(
        "--ee",
        choices=["none", "brainco"],
        default="brainco",
        help="End-effector controller used by --execute.",
    )
    parser.add_argument(
        "--source",
        choices=["actions", "states"],
        default="actions",
        help="Replay actions or recorded states to the robot.",
    )
    parser.add_argument(
        "--initial-target-q",
        "--initial_target_q",
        dest="initial_target_q",
        type=str,
        default=None,
        help='Initial dual-arm joint target, for example "[0, 0, ...]" or "0,0,...".',
    )
    parser.add_argument(
        "--initial-target-q-file",
        "--initial-target-pose-file",
        dest="initial_target_q_file",
        type=Path,
        default=DEFAULT_INITIAL_TARGET_Q_FILE,
        help=f"JSON file containing named initial dual-arm targets. Default: {DEFAULT_INITIAL_TARGET_Q_FILE}",
    )
    parser.add_argument(
        "--initial-target-q-name",
        "--initial-target-pose",
        dest="initial_target_q_name",
        type=str,
        default=None,
        help="Named initial target in the JSON file. Defaults to the episode parent directory name.",
    )
    parser.add_argument(
        "--network-interface",
        type=str,
        default=None,
        help="DDS network interface, for example eth0. Defaults to SDK behavior.",
    )
    parser.add_argument("--sim", action="store_true", help="Use DDS domain 1.")
    parser.add_argument(
        "--motion",
        action="store_true",
        help="Use the arm SDK motion topic instead of debug lowcmd topic.",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=3.0,
        help="Seconds used to ramp from current arm pose to the first replay pose.",
    )
    parser.add_argument(
        "--max-initial-arm-delta",
        type=float,
        default=1.5,
        help="Abort if any arm joint is farther than this from the first target.",
    )
    parser.add_argument(
        "--max-frame-arm-delta",
        type=float,
        default=0.35,
        help="Abort if any adjacent replay arm target changes more than this.",
    )
    parser.add_argument(
        "--allow-large-initial-jump",
        action="store_true",
        help="Bypass the initial arm delta check. Use only after manual inspection.",
    )
    parser.add_argument(
        "--go-home",
        action="store_true",
        help="Call ctrl_dual_arm_go_home() after replay or interruption.",
    )
    return parser.parse_args()


def load_episode(episode_dir: Path) -> dict[str, Any]:
    json_path = episode_dir / "data.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"data.json not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        episode = json.load(f)

    if not isinstance(episode.get("data"), list):
        raise ValueError(f"Invalid episode file: {json_path} has no data list")
    return episode


def selected_frames(
    frames: list[dict[str, Any]], start: int, end: int | None
) -> list[dict[str, Any]]:
    if not frames:
        return []

    first = max(start, 0)
    last = len(frames) - 1 if end is None else min(end, len(frames) - 1)
    if first > last:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")
    return frames[first : last + 1]


def available_cameras(frames: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for frame in frames:
        colors = frame.get("colors") or {}
        if isinstance(colors, dict):
            keys.update(str(key) for key in colors.keys())
    return sorted(keys)


def read_image(episode_dir: Path, relative_path: str) -> np.ndarray | None:
    image_path = episode_dir / str(relative_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[warn] failed to read image: {image_path}", file=sys.stderr)
    return image


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_blank_tile(width: int, height: int, label: str) -> np.ndarray:
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        tile,
        f"{label}: missing",
        (18, max(32, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return tile


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def build_frame_image(
    episode_dir: Path,
    frame: dict[str, Any],
    camera_keys: list[str],
    tile_width: int,
) -> np.ndarray:
    colors = frame.get("colors") or {}
    tiles: list[np.ndarray] = []
    max_height = 1

    for key in camera_keys:
        relative_path = colors.get(key) if isinstance(colors, dict) else None
        image = read_image(episode_dir, relative_path) if relative_path else None
        if image is not None:
            tile = resize_to_width(image, tile_width)
        else:
            tile = make_blank_tile(tile_width, round(tile_width * 0.75), key)
        tile = add_label(tile, key)
        tiles.append(tile)
        max_height = max(max_height, tile.shape[0])

    padded_tiles = []
    for tile in tiles:
        if tile.shape[0] < max_height:
            pad = np.zeros((max_height - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded_tiles.append(tile)

    if not padded_tiles:
        return make_blank_tile(tile_width, round(tile_width * 0.75), "no cameras")
    return np.hstack(padded_tiles)


def qpos_summary(frame: dict[str, Any], kind: str) -> str:
    values = frame.get(kind) or {}
    if not isinstance(values, dict):
        return f"{kind}: none"

    parts = []
    for name in ("left_arm", "right_arm", "left_ee", "right_ee", "body"):
        qpos = (values.get(name) or {}).get("qpos", [])
        if qpos:
            preview = ", ".join(f"{float(v):.3f}" for v in qpos[:3])
            suffix = ", ..." if len(qpos) > 3 else ""
            parts.append(f"{name}[{len(qpos)}]=[{preview}{suffix}]")
    return f"{kind}: " + "; ".join(parts) if parts else f"{kind}: none"


def extract_dual_arm_q(frame: dict[str, Any], source: str) -> np.ndarray | None:
    group = frame.get(source) or {}
    if not isinstance(group, dict):
        return None
    left = (group.get("left_arm") or {}).get("qpos", [])
    right = (group.get("right_arm") or {}).get("qpos", [])
    if len(left) != 7 or len(right) != 7:
        return None
    return np.asarray(left + right, dtype=float)


def extract_dual_hand_q(
    frame: dict[str, Any], source: str
) -> tuple[np.ndarray, np.ndarray] | None:
    group = frame.get(source) or {}
    if not isinstance(group, dict):
        return None
    left = (group.get("left_ee") or {}).get("qpos", [])
    right = (group.get("right_ee") or {}).get("qpos", [])
    if len(left) != 6 or len(right) != 6:
        return None
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float)


def parse_initial_target_q(raw_value: Any, expected_dof: int) -> np.ndarray | None:
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
        raise ValueError(
            f"initial_target_q for this arm must have {expected_dof} values, "
            f"got {len(values)}."
        )
    return np.asarray(values, dtype=float)


def load_named_initial_target_q(
    config_path: Path, pose_name: str | None, arm: str
) -> Any:
    if not pose_name or not config_path.exists():
        return INITIAL_TARGET_Q_MISSING

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse initial target q file {config_path}: {exc}")

    if not isinstance(config, dict):
        raise ValueError(f"initial target q file {config_path} must contain a JSON object.")

    pose_entry = config.get(pose_name)
    if pose_entry is None:
        return INITIAL_TARGET_Q_MISSING
    if isinstance(pose_entry, dict):
        return pose_entry.get(arm, INITIAL_TARGET_Q_MISSING)
    return pose_entry


def resolve_initial_target_q(args: argparse.Namespace, episode_dir: Path) -> np.ndarray | None:
    if args.arm == "none":
        return None

    expected_dof = ARM_TARGET_DOF[args.arm]
    if args.initial_target_q is not None:
        target_q = parse_initial_target_q(args.initial_target_q, expected_dof)
        print("Loaded initial_target_q from --initial-target-q.")
        return target_q

    pose_name = args.initial_target_q_name or episode_dir.parent.name
    config_path = args.initial_target_q_file.expanduser().resolve()
    raw_value = load_named_initial_target_q(config_path, pose_name, args.arm)
    if raw_value is INITIAL_TARGET_Q_MISSING:
        if args.initial_target_q_name is not None:
            raise ValueError(
                f"initial target q name '{args.initial_target_q_name}' for arm "
                f"'{args.arm}' was not found in {config_path}."
            )
        print(
            f"[warn] no initial_target_q '{pose_name}' for {args.arm} in {config_path}; "
            "using controller default.",
            file=sys.stderr,
        )
        return None

    print(f"Loaded initial_target_q '{pose_name}' for {args.arm} from {config_path}.")
    return parse_initial_target_q(raw_value, expected_dof)


def collect_arm_targets(
    frames: list[dict[str, Any]], source: str
) -> list[np.ndarray | None]:
    return [extract_dual_arm_q(frame, source) for frame in frames]


def collect_hand_targets(
    frames: list[dict[str, Any]], source: str
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    return [extract_dual_hand_q(frame, source) for frame in frames]


def max_adjacent_delta(targets: list[np.ndarray | None]) -> float:
    valid = [target for target in targets if target is not None]
    if len(valid) < 2:
        return 0.0
    stacked = np.vstack(valid)
    return float(np.max(np.abs(np.diff(stacked, axis=0))))


class BraincoDirectController:
    """Minimal BrainCo hand publisher for recorded 6-DoF hand q targets."""

    def __init__(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

        self._left_pub = ChannelPublisher("rt/brainco/left/cmd", MotorCmds_)
        self._right_pub = ChannelPublisher("rt/brainco/right/cmd", MotorCmds_)
        self._left_pub.Init()
        self._right_pub.Init()
        self._left_msg = MotorCmds_()
        self._right_msg = MotorCmds_()
        self._left_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        self._right_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
        for msg in (self._left_msg, self._right_msg):
            for cmd in msg.cmds:
                cmd.q = 0.0
                cmd.dq = 1.0

    def ctrl_dual_hand(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        left_q = np.clip(np.asarray(left_q, dtype=float), 0.0, 1.0)
        right_q = np.clip(np.asarray(right_q, dtype=float), 0.0, 1.0)
        if left_q.shape != (6,) or right_q.shape != (6,):
            raise ValueError("BrainCo replay expects 6 q values per hand")
        for idx, value in enumerate(left_q):
            self._left_msg.cmds[idx].q = float(value)
        for idx, value in enumerate(right_q):
            self._right_msg.cmds[idx].q = float(value)
        self._left_pub.Write(self._left_msg)
        self._right_pub.Write(self._right_msg)


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
    print(f"Enter debug mode: {'Success' if status == 0 else 'Failed'} {result}")


def create_arm_controller(
    arm_name: str, motion: bool, sim: bool, initial_target_q: np.ndarray | None
):
    from teleop.robot_control.robot_arm import (
        G1_23_ArmController,
        G1_29_ArmController,
        H1_2_ArmController,
        H1_ArmController,
        H2_ArmController,
    )

    controllers = {
        "G1_29": G1_29_ArmController,
        "G1_23": G1_23_ArmController,
        "H1_2": H1_2_ArmController,
        "H1": H1_ArmController,
        "H2": H2_ArmController,
    }
    cls = controllers[arm_name]
    if arm_name == "H1":
        return cls(simulation_mode=sim, initial_target_q=initial_target_q)
    return cls(motion_mode=motion, simulation_mode=sim, initial_target_q=initial_target_q)


def sleep_to_rate(start_time: float, fps: float, speed: float) -> None:
    period = 1.0 / (fps * speed)
    elapsed = time.time() - start_time
    if elapsed < period:
        time.sleep(period - elapsed)


def execute_robot_replay(
    episode_dir: Path,
    frames: list[dict[str, Any]],
    camera_keys: list[str],
    initial_target_q: np.ndarray | None,
    fps: float,
    speed: float,
    source: str,
    arm_name: str,
    ee_name: str,
    motion: bool,
    sim: bool,
    network_interface: str | None,
    ramp_time: float,
    max_initial_arm_delta: float,
    max_frame_arm_delta: float,
    allow_large_initial_jump: bool,
    go_home: bool,
    tile_width: int,
    show_gui: bool,
) -> None:
    use_arm = arm_name != "none"
    use_hand = ee_name != "none"
    if not use_arm and not use_hand:
        raise ValueError("At least one of --arm or --ee must be enabled for --execute")

    arm_targets = collect_arm_targets(frames, source) if use_arm else []
    hand_targets = collect_hand_targets(frames, source) if use_hand else []

    if use_arm and not any(target is not None for target in arm_targets):
        raise ValueError(f"No 14-DoF arm targets found in {source}")
    if use_hand and not any(target is not None for target in hand_targets):
        raise ValueError(f"No 6-DoF BrainCo hand targets found in {source}")

    frame_delta = max_adjacent_delta(arm_targets)
    if use_arm and frame_delta > max_frame_arm_delta:
        raise ValueError(
            f"Adjacent arm target jump {frame_delta:.3f} exceeds "
            f"--max-frame-arm-delta {max_frame_arm_delta:.3f}"
        )

    print("Initializing DDS and robot controllers...")
    initialize_dds(sim=sim, network_interface=network_interface)
    if use_arm and not motion and not sim:
        enter_debug_mode()

    arm_ctrl = (
        create_arm_controller(
            arm_name, motion=motion, sim=sim, initial_target_q=initial_target_q
        )
        if use_arm
        else None
    )
    hand_ctrl = BraincoDirectController() if use_hand else None
    tau = np.zeros(14, dtype=float)
    if arm_ctrl is not None:
        arm_ctrl.ctrl_dual_arm(arm_ctrl.get_current_dual_arm_q(), tau)
    window_name = f"Episode Execute - {episode_dir.name}"

    try:
        if arm_ctrl is not None:
            first_arm_target = next(target for target in arm_targets if target is not None)
            current_arm_q = arm_ctrl.get_current_dual_arm_q()
            ramp_start_q = current_arm_q
            if initial_target_q is not None:
                initial_pose_delta = float(np.max(np.abs(initial_target_q - current_arm_q)))
                print(f"current-to-configured-initial max arm delta: {initial_pose_delta:.3f} rad")
                if initial_pose_delta > max_initial_arm_delta and not allow_large_initial_jump:
                    raise ValueError(
                        f"Configured initial pose is too far from current pose: "
                        f"{initial_pose_delta:.3f} > {max_initial_arm_delta:.3f}. "
                        "Move closer first or use --allow-large-initial-jump after inspection."
                    )

                initial_ramp_steps = max(1, int(max(ramp_time, 0.0) * fps))
                print(f"Ramping arm to configured initial pose in {initial_ramp_steps} steps...")
                for step in range(1, initial_ramp_steps + 1):
                    loop_start = time.time()
                    ratio = step / initial_ramp_steps
                    target = current_arm_q + (initial_target_q - current_arm_q) * ratio
                    arm_ctrl.ctrl_dual_arm(target, tau)
                    sleep_to_rate(loop_start, fps=fps, speed=1.0)
                ramp_start_q = arm_ctrl.get_current_dual_arm_q()

            initial_delta = float(np.max(np.abs(first_arm_target - ramp_start_q)))
            if initial_target_q is not None:
                print(f"configured-initial-to-first max arm delta: {initial_delta:.3f} rad")
            else:
                print(f"current-to-first max arm delta: {initial_delta:.3f} rad")
            if initial_delta > max_initial_arm_delta and not allow_large_initial_jump:
                raise ValueError(
                    f"Initial arm target is too far from replay start pose: {initial_delta:.3f} "
                    f"> {max_initial_arm_delta:.3f}. Move closer first or use "
                    "--allow-large-initial-jump after inspection."
                )

            ramp_steps = max(1, int(max(ramp_time, 0.0) * fps))
            print(f"Ramping arm to first target in {ramp_steps} steps...")
            for step in range(1, ramp_steps + 1):
                loop_start = time.time()
                ratio = step / ramp_steps
                target = ramp_start_q + (first_arm_target - ramp_start_q) * ratio
                arm_ctrl.ctrl_dual_arm(target, tau)
                sleep_to_rate(loop_start, fps=fps, speed=1.0)

        if hand_ctrl is not None:
            first_hand_target = next(target for target in hand_targets if target is not None)
            hand_ctrl.ctrl_dual_hand(*first_hand_target)

        print(
            f"Executing {len(frames)} frames at {fps:g} FPS, speed={speed:g}, "
            f"source={source}."
        )
        for pos, frame in enumerate(frames):
            loop_start = time.time()
            if arm_ctrl is not None:
                arm_target = arm_targets[pos]
                if arm_target is not None:
                    arm_ctrl.ctrl_dual_arm(arm_target, tau)
            if hand_ctrl is not None:
                hand_target = hand_targets[pos]
                if hand_target is not None:
                    hand_ctrl.ctrl_dual_hand(*hand_target)
            if show_gui and camera_keys:
                frame_image = build_frame_image(
                    episode_dir=episode_dir,
                    frame=frame,
                    camera_keys=camera_keys,
                    tile_width=tile_width,
                )
                idx = frame.get("idx", pos)
                title = (
                    f"EXECUTE idx={idx} frame={pos + 1}/{len(frames)} "
                    f"fps={fps:g} speed={speed:g} source={source}"
                )
                cv2.putText(
                    frame_image,
                    title,
                    (10, frame_image.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame_image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopped by user.")
                    return
                if key == ord(" "):
                    while True:
                        pause_key = cv2.waitKey(0) & 0xFF
                        if pause_key in (ord(" "), ord("q"), 27):
                            break
                    if pause_key in (ord("q"), 27):
                        print("Stopped by user.")
                        return
            if pos == 0 or (pos + 1) % 30 == 0 or pos == len(frames) - 1:
                print(f"sent frame {pos + 1}/{len(frames)} idx={frame.get('idx', pos)}")
            sleep_to_rate(loop_start, fps=fps, speed=speed)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if go_home and arm_ctrl is not None:
            print("Returning arms to home...")
            arm_ctrl.ctrl_dual_arm_go_home()
        if show_gui:
            cv2.destroyAllWindows()


def print_episode_summary(
    episode_dir: Path, episode: dict[str, Any], frames: list[dict[str, Any]]
) -> None:
    info = episode.get("info") or {}
    text = episode.get("text") or {}
    fps = (info.get("image") or {}).get("fps", "unknown")
    cameras = available_cameras(frames)

    print(f"episode: {episode_dir}")
    print(f"frames: {len(frames)}")
    print(f"fps: {fps}")
    print(f"cameras: {', '.join(cameras) if cameras else 'none'}")
    if text.get("goal"):
        print(f"goal: {text['goal']}")
    if text.get("desc"):
        print(f"desc: {text['desc']}")


def has_display() -> bool:
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_video_writer(path: Path, fps: float, frame_image: np.ndarray) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frame_image.shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def replay(
    episode_dir: Path,
    frames: list[dict[str, Any]],
    camera_keys: list[str],
    fps: float,
    speed: float,
    tile_width: int,
    print_every: int,
    show_gui: bool,
    export_video: Path | None,
    loop: bool,
) -> None:
    delay = 0.0 if fps <= 0 else 1.0 / (fps * max(speed, 1e-6))
    writer: cv2.VideoWriter | None = None
    window_name = f"Episode Replay - {episode_dir.name}"

    try:
        while True:
            for pos, frame in enumerate(frames):
                frame_image = build_frame_image(
                    episode_dir=episode_dir,
                    frame=frame,
                    camera_keys=camera_keys,
                    tile_width=tile_width,
                )

                idx = frame.get("idx", pos)
                title = f"idx={idx} frame={pos + 1}/{len(frames)} fps={fps:g} speed={speed:g}"
                cv2.putText(
                    frame_image,
                    title,
                    (10, frame_image.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if writer is None and export_video is not None:
                    writer = open_video_writer(export_video, fps, frame_image)
                if writer is not None:
                    writer.write(frame_image)

                if show_gui:
                    cv2.imshow(window_name, frame_image)
                    key = cv2.waitKey(max(1, round(delay * 1000))) & 0xFF
                    if key in (ord("q"), 27):
                        return
                    if key == ord(" "):
                        while True:
                            pause_key = cv2.waitKey(0) & 0xFF
                            if pause_key in (ord(" "), ord("q"), 27):
                                break
                        if pause_key in (ord("q"), 27):
                            return
                elif delay > 0 and export_video is None:
                    time.sleep(delay)

                if print_every > 0 and (pos == 0 or (pos + 1) % print_every == 0):
                    print(f"\nidx={idx} frame={pos + 1}/{len(frames)}")
                    print(qpos_summary(frame, "states"))
                    print(qpos_summary(frame, "actions"))

            if not loop:
                return
    finally:
        if writer is not None:
            writer.release()
            print(f"video saved: {export_video}")
        if show_gui:
            cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    episode_dir = args.episode_dir.expanduser().resolve()

    try:
        episode = load_episode(episode_dir)
        all_frames = episode["data"]
        frames = selected_frames(all_frames, args.start, args.end)
        if not frames:
            raise ValueError(f"No frames found in episode: {episode_dir}")
        cameras = available_cameras(frames)
        camera_keys = args.camera or cameras

        image_info = (episode.get("info") or {}).get("image") or {}
        fps = args.fps if args.fps is not None else float(image_info.get("fps", 30.0))
        if fps <= 0:
            raise ValueError("--fps must be positive")
        if args.speed <= 0:
            raise ValueError("--speed must be positive")

        print_episode_summary(episode_dir, episode, all_frames)
        print(f"selected frames: {frames[0].get('idx', 0)}..{frames[-1].get('idx', len(frames)-1)}")
        print(f"selected cameras: {', '.join(camera_keys) if camera_keys else 'none'}")

        if args.execute:
            if not args.yes:
                raise ValueError(
                    "--execute will move the physical robot. Re-run with --yes "
                    "after confirming the workspace is clear."
                )
            show_gui = not args.no_gui
            if show_gui and not has_display():
                print("[warn] no display found; switching to --no-gui mode", file=sys.stderr)
                show_gui = False
            if show_gui:
                print("keys: q or Esc to stop, Space to pause/resume")
            initial_target_q = resolve_initial_target_q(args, episode_dir)
            execute_robot_replay(
                episode_dir=episode_dir,
                frames=frames,
                camera_keys=camera_keys,
                initial_target_q=initial_target_q,
                fps=fps,
                speed=args.speed,
                source=args.source,
                arm_name=args.arm,
                ee_name=args.ee,
                motion=args.motion,
                sim=args.sim,
                network_interface=args.network_interface,
                ramp_time=args.ramp_time,
                max_initial_arm_delta=args.max_initial_arm_delta,
                max_frame_arm_delta=args.max_frame_arm_delta,
                allow_large_initial_jump=args.allow_large_initial_jump,
                go_home=args.go_home,
                tile_width=args.tile_width,
                show_gui=show_gui,
            )
            return 0

        missing = [key for key in camera_keys if key not in cameras]
        if missing:
            raise ValueError(
                f"Camera keys not found: {', '.join(missing)}. "
                f"Available: {', '.join(cameras)}"
            )
        if not camera_keys:
            raise ValueError("No color cameras found in the selected frame range.")

        print("keys: q or Esc to quit, Space to pause/resume")

        show_gui = not args.no_gui
        if show_gui and not has_display():
            print("[warn] no display found; switching to --no-gui mode", file=sys.stderr)
            show_gui = False

        replay(
            episode_dir=episode_dir,
            frames=frames,
            camera_keys=camera_keys,
            fps=fps,
            speed=args.speed,
            tile_width=args.tile_width,
            print_every=args.print_every,
            show_gui=show_gui,
            export_video=args.export_video,
            loop=args.loop,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
