#!/usr/bin/env python3
"""Replace vehicle button episode text by task id from button_subtasks.json.

List available tasks:
    python3 fix_vehicle_button_text_by_task.py --list-tasks

Dry run:
    python3 fix_vehicle_button_text_by_task.py --start 1 --end 56 --task-id 1 --dry-run

Apply:
    python3 fix_vehicle_button_text_by_task.py --start 1 --end 56 --task-id 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fix_vehicle_button_text as fixer


DEFAULT_DATASET_DIR = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press_ccw"
)
DEFAULT_SUBTASKS_PATH = Path(
    "/home/ubuntu/zty/xr_teleoperate_shu/teleop/button_subtasks.json"
)
DEFAULT_TASK_GROUP = "vehicle_physical_button_press_ccw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace data.json -> text for a selected episode range using a "
            "task definition from button_subtasks.json."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--subtasks-json",
        type=Path,
        default=DEFAULT_SUBTASKS_PATH,
        help=f"Task language JSON file. Default: {DEFAULT_SUBTASKS_PATH}",
    )
    parser.add_argument(
        "--task-group",
        default=DEFAULT_TASK_GROUP,
        help=f"Top-level task group in subtasks JSON. Default: {DEFAULT_TASK_GROUP}",
    )
    parser.add_argument(
        "--task-id",
        help=(
            "Task id to use. Supports 1-based numeric id, such as 1-6, "
            "or string id, such as front_windshield_defrost."
        ),
    )
    parser.add_argument(
        "--min-episode",
        "--start",
        dest="min_episode",
        type=int,
        help="Replace episodes with number >= this value.",
    )
    parser.add_argument(
        "--max-episode",
        "--end",
        dest="max_episode",
        type=int,
        help="Replace episodes with number <= this value.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available tasks and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching files without writing changes.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write a .bak file next to each changed data.json before replacing it.",
    )
    return parser.parse_args()


def load_tasks(subtasks_path: Path, task_group: str) -> list[dict[str, Any]]:
    with subtasks_path.open("r", encoding="utf-8") as f:
        subtasks = json.load(f)

    tasks = subtasks.get(task_group)
    if not isinstance(tasks, list):
        available = ", ".join(sorted(subtasks.keys()))
        raise KeyError(
            f"Task group not found: {task_group}. Available groups: {available}"
        )
    return tasks


def task_text(task: dict[str, Any]) -> dict[str, str]:
    required_keys = ("goal", "desc", "steps")
    missing = [key for key in required_keys if key not in task]
    if missing:
        task_id = task.get("id", "<unknown>")
        raise KeyError(f"Task {task_id} is missing keys: {', '.join(missing)}")
    return {key: str(task[key]) for key in required_keys}


def resolve_task(tasks: list[dict[str, Any]], task_id: str) -> tuple[int, dict[str, Any]]:
    if task_id.isdigit():
        index = int(task_id)
        if 1 <= index <= len(tasks):
            return index, tasks[index - 1]
        raise ValueError(f"Numeric task id must be in 1-{len(tasks)}: {task_id}")

    for index, task in enumerate(tasks, 1):
        if task.get("id") == task_id:
            return index, task
    valid_ids = ", ".join(str(task.get("id")) for task in tasks)
    raise ValueError(f"Unknown task id: {task_id}. Valid ids: {valid_ids}")


def print_tasks(tasks: list[dict[str, Any]]) -> None:
    print("Available tasks:")
    for index, task in enumerate(tasks, 1):
        print(f"{index}. {task.get('id')}: {task.get('goal')}")


def format_episode(number: int) -> str:
    return f"episode_{number:04d}"


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


def format_ranges(numbers: list[int]) -> str:
    parts = []
    for start, end in compact_ranges(numbers):
        if start == end:
            parts.append(format_episode(start))
        else:
            parts.append(f"{format_episode(start)}-{format_episode(end)}")
    return ", ".join(parts)


def normalize_text(text_data: Any) -> dict[str, str]:
    if not isinstance(text_data, dict):
        return {"goal": str(text_data), "desc": "", "steps": ""}
    return {
        "goal": str(text_data.get("goal", "")),
        "desc": str(text_data.get("desc", "")),
        "steps": str(text_data.get("steps", "")),
    }


def text_key(text_data: dict[str, str]) -> tuple[str, str, str]:
    return text_data["goal"], text_data["desc"], text_data["steps"]


def print_current_text_summary(
    files: list[tuple[int, Path]],
    target_text: dict[str, str],
) -> None:
    groups: dict[tuple[str, str, str], list[int]] = {}
    errors: list[tuple[int, Path, str]] = []

    for number, data_path in files:
        try:
            with data_path.open("r", encoding="utf-8") as f:
                episode_data = json.load(f)
            raw_text = episode_data.get("text") if isinstance(episode_data, dict) else None
            current_text = normalize_text(raw_text)
        except Exception as exc:
            errors.append((number, data_path, repr(exc)))
            continue

        groups.setdefault(text_key(current_text), []).append(number)

    print("Current task language(s) in selected range:")
    for index, (key, episodes) in enumerate(
        sorted(groups.items(), key=lambda item: min(item[1])),
        1,
    ):
        current_text = {"goal": key[0], "desc": key[1], "steps": key[2]}
        same_as_target = current_text == target_text
        status = "same as target" if same_as_target else "will be replaced"
        print(f"  {index}. {status}")
        print(f"     count: {len(episodes)}  episodes: {format_ranges(episodes)}")
        print(f"     goal: {current_text['goal']}")
        if current_text["desc"]:
            print(f"     desc: {current_text['desc']}")

    if errors:
        print("  Read errors while checking current language:")
        for number, data_path, error in errors:
            print(f"     {format_episode(number)}: {data_path} ({error})")


def print_target_task(index: int, task: dict[str, Any]) -> None:
    print("Replacement task language:")
    print(f"  task_id: {index} ({task.get('id')})")
    print(f"  goal: {task.get('goal')}")
    print(f"  desc: {task.get('desc')}")
    print(f"  steps: {task.get('steps')}")


def validate_args(args: argparse.Namespace) -> None:
    if args.list_tasks:
        return
    if args.task_id is None:
        raise ValueError("Please provide --task-id, for example --task-id 1")
    if args.min_episode is None or args.max_episode is None:
        raise ValueError("Please provide --start and --end episode numbers")
    if args.min_episode > args.max_episode:
        raise ValueError(
            f"min episode must be <= max episode: "
            f"{args.min_episode} > {args.max_episode}"
        )


def main() -> int:
    args = parse_args()
    validate_args(args)

    dataset_dir = args.dataset_dir.expanduser()
    subtasks_path = args.subtasks_json.expanduser()
    if not subtasks_path.is_file():
        raise FileNotFoundError(f"Subtasks JSON not found: {subtasks_path}")

    tasks = load_tasks(subtasks_path, args.task_group)
    if args.list_tasks:
        print_tasks(tasks)
        return 0

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    task_index, task = resolve_task(tasks, args.task_id)
    target_text = task_text(task)
    fixer.TARGET_TEXT = target_text

    files = fixer.iter_episode_data_files(
        dataset_dir,
        args.min_episode,
        args.max_episode,
    )
    if not files:
        print("No matching episode data.json files found.")
        return 0

    print(f"Episode range: {format_episode(args.min_episode)}-{format_episode(args.max_episode)}")
    print()
    print_current_text_summary(files, target_text)
    print()
    print_target_task(task_index, task)
    print()

    changed = 0
    unchanged = 0
    failed = 0
    for number, data_path in files:
        try:
            raw = data_path.read_text(encoding="utf-8")
            updated, needs_write = fixer.replace_text_field(raw)
        except Exception as exc:
            failed += 1
            print(f"FAILED episode_{number:04d}: {data_path} ({exc!r})")
            continue

        if not needs_write:
            unchanged += 1
            print(f"UNCHANGED episode_{number:04d}: {data_path}")
            continue

        changed += 1
        if args.dry_run:
            print(f"WOULD UPDATE episode_{number:04d}: {data_path}")
            continue

        if args.backup:
            backup_path = data_path.with_suffix(data_path.suffix + ".bak")
            backup_path.write_text(raw, encoding="utf-8")
        fixer.write_atomic(data_path, updated)
        print(f"UPDATED episode_{number:04d}: {data_path}")

    mode = "would update" if args.dry_run else "updated"
    print(
        f"Done: {changed} {mode}, {unchanged} unchanged, "
        f"{failed} failed, {len(files)} checked."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
