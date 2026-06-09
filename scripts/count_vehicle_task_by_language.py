#!/usr/bin/env python3
"""Count vehicle physical button episodes grouped by task language text.

Default:
    python3 count_vehicle_task_by_language.py

Show more detail:
    python3 count_vehicle_task_by_language.py --show-ranges --show-frames

Specify another dataset:
    python3 count_vehicle_task_by_language.py --dataset-dir /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press_ccw"
)
EPISODE_RE = re.compile(r"episode_(\d+)$")


@dataclass
class TaskStats:
    language: str
    goal: str
    desc: str
    steps: str
    episodes: list[int] = field(default_factory=list)
    frame_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count episode data by different task language text and print "
            "episode count plus start/end episode numbers."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--show-desc",
        action="store_true",
        help="Also print desc and steps for each task group.",
    )
    parser.add_argument(
        "--show-ranges",
        action="store_true",
        help="Also print all continuous episode ranges for each task group.",
    )
    parser.add_argument(
        "--show-frames",
        action="store_true",
        help="Also print total frame count for each task group.",
    )
    return parser.parse_args()


def episode_number(path: Path) -> int | None:
    match = EPISODE_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1))


def detect_language(text: str) -> str:
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
    has_alpha = any(("a" <= char.lower() <= "z") for char in text)
    if has_cjk and has_alpha:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_alpha:
        return "en"
    return "unknown"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def make_task_key(text_data: Any) -> tuple[str, str, str, str]:
    if not isinstance(text_data, dict):
        raw_text = as_text(text_data)
        return detect_language(raw_text), raw_text, "", ""

    goal = as_text(text_data.get("goal"))
    desc = as_text(text_data.get("desc"))
    steps = as_text(text_data.get("steps"))
    combined = "\n".join(part for part in (goal, desc, steps) if part)
    return detect_language(combined), goal, desc, steps


def iter_episode_data_files(dataset_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for episode_dir in dataset_dir.iterdir():
        if not episode_dir.is_dir():
            continue
        number = episode_number(episode_dir)
        if number is None:
            continue
        data_path = episode_dir / "data.json"
        if data_path.exists():
            files.append((number, data_path))
    return sorted(files)


def compact_ranges(numbers: list[int]) -> list[tuple[int, int]]:
    if not numbers:
        return []

    sorted_numbers = sorted(numbers)
    ranges: list[tuple[int, int]] = []
    start = previous = sorted_numbers[0]
    for number in sorted_numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous))
        start = previous = number
    ranges.append((start, previous))
    return ranges


def format_episode(number: int) -> str:
    return f"{number:04d}"


def format_ranges(numbers: list[int]) -> str:
    parts = []
    for start, end in compact_ranges(numbers):
        if start == end:
            parts.append(format_episode(start))
        else:
            parts.append(f"{format_episode(start)}-{format_episode(end)}")
    return ", ".join(parts)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    groups: dict[tuple[str, str, str, str], TaskStats] = {}
    errors: list[tuple[int, Path, str]] = []
    files = iter_episode_data_files(dataset_dir)

    for episode, data_path in files:
        try:
            with data_path.open("r", encoding="utf-8") as f:
                episode_data = json.load(f)
        except Exception as exc:
            errors.append((episode, data_path, repr(exc)))
            continue

        text_data = episode_data.get("text") if isinstance(episode_data, dict) else None
        key = make_task_key(text_data)
        if key not in groups:
            language, goal, desc, steps = key
            groups[key] = TaskStats(
                language=language,
                goal=goal,
                desc=desc,
                steps=steps,
            )

        stats = groups[key]
        stats.episodes.append(episode)
        data_items = episode_data.get("data") if isinstance(episode_data, dict) else None
        if isinstance(data_items, list):
            stats.frame_count += len(data_items)

    sorted_groups = sorted(
        groups.values(),
        key=lambda stats: (min(stats.episodes), stats.goal),
    )

    valid_count = sum(len(stats.episodes) for stats in sorted_groups)
    print(f"Dataset: {dataset_dir}")
    print(f"Total: {valid_count}/{len(files)} episodes, {len(sorted_groups)} tasks")
    print()

    for index, stats in enumerate(sorted_groups, 1):
        episodes = sorted(stats.episodes)
        start = episodes[0]
        end = episodes[-1]
        print(f"{index}. {stats.goal}")
        detail = (
            f"   count: {len(episodes)}  "
            f"episodes: {format_episode(start)}-{format_episode(end)}  "
            f"lang: {stats.language}"
        )
        if args.show_frames:
            detail += f"  frames: {stats.frame_count}"
        print(detail)
        if args.show_ranges:
            print(f"   ranges: {format_ranges(episodes)}")
        if args.show_desc:
            if stats.desc:
                print(f"     desc: {stats.desc}")
            if stats.steps:
                print(f"     steps: {stats.steps}")

    if errors:
        print()
        print("Read errors:")
        for episode, data_path, error in errors:
            print(f"  episode_{format_episode(episode)}  {data_path}: {error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
