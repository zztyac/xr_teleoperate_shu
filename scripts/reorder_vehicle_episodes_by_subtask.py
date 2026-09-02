#!/usr/bin/env python3
"""Group episode ids into continuous task intervals ordered by button_subtasks.json.

Dry run:
    python3 reorder_vehicle_episodes_by_subtask.py

Apply directory renames:
    python3 reorder_vehicle_episodes_by_subtask.py --apply

Write a mapping file while applying or dry-running:
    python3 reorder_vehicle_episodes_by_subtask.py --mapping-json /tmp/episode_mapping.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press_ccw"
)
DEFAULT_SUBTASKS_JSON = Path(
    "/home/ubuntu/zty/xr_teleoperate_shu/teleop/button_subtasks.json"
)
DEFAULT_TASK_GROUP = "vehicle_physical_button_press_ccw"
EPISODE_RE = re.compile(r"episode_(\d+)$")
TEMP_PREFIX = ".reorder_tmp_"


@dataclass(frozen=True)
class TaskDef:
    index: int
    task_id: str
    language: str
    goal: str
    desc: str
    steps: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.goal, self.desc, self.steps


@dataclass(frozen=True)
class EpisodeInfo:
    number: int
    path: Path
    data_path: Path
    task: TaskDef
    language: str


@dataclass(frozen=True)
class RenameItem:
    episode: EpisodeInfo
    new_number: int
    target: Path
    temp: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename episode directories so episodes with the same language and "
            "task occupy continuous id intervals. Task interval order follows "
            "button_subtasks.json."
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
        default=DEFAULT_SUBTASKS_JSON,
        help=f"Task definition JSON. Default: {DEFAULT_SUBTASKS_JSON}",
    )
    parser.add_argument(
        "--task-group",
        default=DEFAULT_TASK_GROUP,
        help=f"Top-level key in subtasks JSON. Default: {DEFAULT_TASK_GROUP}",
    )
    parser.add_argument(
        "--language",
        help=(
            "Only reorder this detected language, for example en or zh. "
            "By default all detected languages are reordered, language groups "
            "ordered by first appearance."
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First target episode number. Default: 0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4,
        help="Zero-padding width for episode directory names. Default: 4",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=120,
        help="Maximum old->new mappings to print. Default: 120",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        help="Optional path to write the complete old->new mapping as JSON.",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Skip episodes whose text does not exactly match a task definition.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename directories. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def detect_language(text: str) -> str:
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
    has_alpha = any("a" <= char.lower() <= "z" for char in text)
    if has_cjk and has_alpha:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_alpha:
        return "en"
    return "unknown"


def episode_number(path: Path) -> int | None:
    match = EPISODE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def episode_name(number: int, width: int) -> str:
    return f"episode_{number:0{width}d}"


def load_tasks(subtasks_path: Path, task_group: str) -> list[TaskDef]:
    with subtasks_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    task_items = raw.get(task_group)
    if not isinstance(task_items, list):
        available = ", ".join(sorted(str(key) for key in raw.keys()))
        raise KeyError(
            f"Task group not found: {task_group}. Available groups: {available}"
        )

    tasks: list[TaskDef] = []
    for index, task in enumerate(task_items):
        if not isinstance(task, dict):
            raise TypeError(f"Task #{index + 1} is not an object: {task!r}")
        goal = as_text(task.get("goal"))
        desc = as_text(task.get("desc"))
        steps = as_text(task.get("steps"))
        task_id = as_text(task.get("id")) or str(index + 1)
        if not goal:
            raise ValueError(f"Task #{index + 1} ({task_id}) has empty goal")
        language = detect_language("\n".join((goal, desc, steps)))
        tasks.append(
            TaskDef(
                index=index,
                task_id=task_id,
                language=language,
                goal=goal,
                desc=desc,
                steps=steps,
            )
        )
    return tasks


def load_episode_text(data_path: Path) -> tuple[str, str, str]:
    with data_path.open("r", encoding="utf-8") as f:
        episode_data = json.load(f)
    if not isinstance(episode_data, dict):
        raise TypeError("data.json top-level value is not an object")
    text = episode_data.get("text")
    if not isinstance(text, dict):
        return as_text(text), "", ""
    return (
        as_text(text.get("goal")),
        as_text(text.get("desc")),
        as_text(text.get("steps")),
    )


def iter_episode_dirs(dataset_dir: Path) -> list[tuple[int, Path, Path]]:
    episodes: list[tuple[int, Path, Path]] = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        number = episode_number(child)
        if number is None:
            continue
        data_path = child / "data.json"
        if data_path.is_file():
            episodes.append((number, child, data_path))
    return sorted(episodes)


def build_task_lookup(tasks: list[TaskDef]) -> dict[tuple[str, str, str], TaskDef]:
    lookup: dict[tuple[str, str, str], TaskDef] = {}
    for task in tasks:
        if task.key in lookup:
            previous = lookup[task.key]
            raise ValueError(
                "Duplicate task text in subtasks JSON: "
                f"{previous.task_id} and {task.task_id}"
            )
        lookup[task.key] = task
    return lookup


def collect_episodes(
    dataset_dir: Path,
    tasks: list[TaskDef],
    language_filter: str | None,
    allow_unmatched: bool,
) -> tuple[list[EpisodeInfo], list[str]]:
    task_lookup = build_task_lookup(tasks)
    episodes: list[EpisodeInfo] = []
    skipped: list[str] = []

    for number, episode_dir, data_path in iter_episode_dirs(dataset_dir):
        text_key = load_episode_text(data_path)
        task = task_lookup.get(text_key)
        language = detect_language("\n".join(text_key))
        if task is None:
            message = (
                f"{episode_name(number, 4)} text does not match any task: "
                f"goal={text_key[0]!r}"
            )
            if allow_unmatched:
                skipped.append(message)
                continue
            raise ValueError(message)
        if language_filter is not None and language != language_filter:
            skipped.append(
                f"{episode_name(number, 4)} skipped by language filter "
                f"({language} != {language_filter})"
            )
            continue
        episodes.append(
            EpisodeInfo(
                number=number,
                path=episode_dir,
                data_path=data_path,
                task=task,
                language=language,
            )
        )

    return episodes, skipped


def language_order(episodes: list[EpisodeInfo]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for episode in sorted(episodes, key=lambda item: item.number):
        if episode.language in seen:
            continue
        seen.add(episode.language)
        ordered.append(episode.language)
    return ordered


def build_ordered_episodes(
    episodes: list[EpisodeInfo],
    tasks: list[TaskDef],
) -> list[EpisodeInfo]:
    by_language_task: dict[tuple[str, int], list[EpisodeInfo]] = {}
    for episode in episodes:
        key = (episode.language, episode.task.index)
        by_language_task.setdefault(key, []).append(episode)

    ordered: list[EpisodeInfo] = []
    for language in language_order(episodes):
        for task in tasks:
            task_episodes = by_language_task.get((language, task.index), [])
            ordered.extend(sorted(task_episodes, key=lambda item: item.number))
    return ordered


def temp_path(dataset_dir: Path, index: int, source: Path) -> Path:
    return dataset_dir / f"{TEMP_PREFIX}{index:06d}_{source.name}"


def build_plan(
    dataset_dir: Path,
    ordered_episodes: list[EpisodeInfo],
    start: int,
    width: int,
) -> list[RenameItem]:
    source_paths = {episode.path for episode in ordered_episodes}
    target_paths = [
        dataset_dir / episode_name(start + index, width)
        for index, _ in enumerate(ordered_episodes)
    ]

    duplicate_targets = sorted(
        {path for path in target_paths if target_paths.count(path) > 1}
    )
    if duplicate_targets:
        names = ", ".join(path.name for path in duplicate_targets)
        raise ValueError(f"Duplicate target names generated: {names}")

    conflicts = [
        path for path in target_paths if path.exists() and path not in source_paths
    ]
    if conflicts:
        names = ", ".join(path.name for path in conflicts[:20])
        suffix = "" if len(conflicts) <= 20 else f", ... {len(conflicts) - 20} more"
        raise FileExistsError(
            "Target already exists but is not in the selected episode set: "
            f"{names}{suffix}"
        )

    plan: list[RenameItem] = []
    for index, (episode, target) in enumerate(zip(ordered_episodes, target_paths), 1):
        if episode.path == target:
            continue
        tmp = temp_path(dataset_dir, index, episode.path)
        if tmp.exists():
            raise FileExistsError(f"Temporary path already exists: {tmp}")
        plan.append(
            RenameItem(
                episode=episode,
                new_number=start + index - 1,
                target=target,
                temp=tmp,
            )
        )
    return plan


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


def format_ranges(numbers: list[int], width: int) -> str:
    parts: list[str] = []
    for start, end in compact_ranges(numbers):
        if start == end:
            parts.append(f"{start:0{width}d}")
        else:
            parts.append(f"{start:0{width}d}-{end:0{width}d}")
    return ", ".join(parts)


def print_summary(
    ordered_episodes: list[EpisodeInfo],
    tasks: list[TaskDef],
    start: int,
    width: int,
) -> None:
    offset = start
    print("Target intervals:")
    for language in language_order(ordered_episodes):
        print(f"  language: {language}")
        for task in tasks:
            task_episodes = [
                episode
                for episode in ordered_episodes
                if episode.language == language and episode.task.index == task.index
            ]
            if not task_episodes:
                continue
            count = len(task_episodes)
            new_numbers = list(range(offset, offset + count))
            old_numbers = [episode.number for episode in task_episodes]
            print(
                f"    {task.index + 1}. {task.task_id}: count={count}  "
                f"old={format_ranges(old_numbers, width)}  "
                f"new={format_ranges(new_numbers, width)}"
            )
            offset += count


def mapping_records(plan_source: list[EpisodeInfo], start: int, width: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, episode in enumerate(plan_source):
        new_number = start + index
        records.append(
            {
                "old_episode": episode_name(episode.number, width),
                "new_episode": episode_name(new_number, width),
                "old_number": episode.number,
                "new_number": new_number,
                "language": episode.language,
                "task_index": episode.task.index + 1,
                "task_id": episode.task.task_id,
                "goal": episode.task.goal,
            }
        )
    return records


def print_plan(
    dataset_dir: Path,
    ordered_episodes: list[EpisodeInfo],
    rename_items: list[RenameItem],
    start: int,
    width: int,
    preview: int,
    apply: bool,
    skipped: list[str],
) -> None:
    mode = "APPLY" if apply else "DRY RUN"
    end = start + len(ordered_episodes) - 1
    print(f"Mode: {mode}")
    print(f"Dataset: {dataset_dir}")
    print(f"Selected episodes: {len(ordered_episodes)}")
    print(
        f"Target sequence: {episode_name(start, width)}-{episode_name(end, width)}"
        if ordered_episodes
        else "Target sequence: <empty>"
    )
    print(f"Need rename: {len(rename_items)}")
    if skipped:
        print(f"Skipped episodes: {len(skipped)}")
    print()

    if ordered_episodes:
        print_summary(ordered_episodes, sorted({ep.task for ep in ordered_episodes}, key=lambda task: task.index), start, width)
        print()

    if rename_items:
        print("Rename plan:")
        for item in rename_items[:preview]:
            print(
                f"  {item.episode.path.name} -> {item.target.name}  "
                f"({item.episode.language}, {item.episode.task.task_id})"
            )
        hidden = len(rename_items) - min(len(rename_items), preview)
        if hidden:
            print(f"  ... {hidden} more")
        print()

    if skipped and preview:
        print("Skipped preview:")
        for message in skipped[: min(len(skipped), preview)]:
            print(f"  {message}")
        hidden = len(skipped) - min(len(skipped), preview)
        if hidden:
            print(f"  ... {hidden} more")
        print()

    if not apply:
        print("No files changed. Add --apply to rename directories.")


def write_mapping(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_plan(rename_items: list[RenameItem]) -> None:
    for item in rename_items:
        item.episode.path.rename(item.temp)

    for item in rename_items:
        if item.target.exists():
            raise FileExistsError(f"Target exists before final rename: {item.target}")
        item.temp.rename(item.target)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    subtasks_path = args.subtasks_json.expanduser()
    if args.start < 0:
        raise ValueError(f"--start must be >= 0: {args.start}")
    if args.width < 1:
        raise ValueError(f"--width must be >= 1: {args.width}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not subtasks_path.is_file():
        raise FileNotFoundError(f"Subtasks JSON not found: {subtasks_path}")

    tasks = load_tasks(subtasks_path, args.task_group)
    selected, skipped = collect_episodes(
        dataset_dir=dataset_dir,
        tasks=tasks,
        language_filter=args.language,
        allow_unmatched=args.allow_unmatched,
    )
    ordered = build_ordered_episodes(selected, tasks)
    rename_items = build_plan(dataset_dir, ordered, args.start, args.width)
    print_plan(
        dataset_dir=dataset_dir,
        ordered_episodes=ordered,
        rename_items=rename_items,
        start=args.start,
        width=args.width,
        preview=args.preview,
        apply=args.apply,
        skipped=skipped,
    )

    if args.mapping_json:
        mapping_path = args.mapping_json.expanduser()
        write_mapping(mapping_path, mapping_records(ordered, args.start, args.width))
        print(f"Wrote mapping: {mapping_path}")

    if args.apply and rename_items:
        apply_plan(rename_items)
        print(f"Renamed {len(rename_items)} episode directories.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
