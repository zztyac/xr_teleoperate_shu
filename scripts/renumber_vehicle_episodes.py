#!/usr/bin/env python3
"""Renumber episode directories sequentially from episode_0001.

Dry run by default:
    python3 renumber_vehicle_episodes.py

Apply changes:
    python3 renumber_vehicle_episodes.py --apply
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_DIR = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press_ccw"
)
EPISODE_RE = re.compile(r"episode_(\d+)$")
TEMP_PREFIX = ".renumber_tmp_"


@dataclass(frozen=True)
class EpisodeDir:
    number: int
    path: Path


@dataclass(frozen=True)
class RenameItem:
    source: Path
    target: Path
    temp: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename episode directories in numeric order to a continuous "
            "sequence such as episode_0001, episode_0002, ..."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First episode number after renaming. Default: 1",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4,
        help="Zero-padding width for episode names. Default: 4",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename directories. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=80,
        help="Maximum rename mappings to print. Default: 80",
    )
    return parser.parse_args()


def episode_number(path: Path) -> int | None:
    match = EPISODE_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1))


def iter_episode_dirs(dataset_dir: Path) -> list[EpisodeDir]:
    episodes: list[EpisodeDir] = []
    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        number = episode_number(child)
        if number is None:
            continue
        episodes.append(EpisodeDir(number=number, path=child))
    return sorted(episodes, key=lambda item: (item.number, item.path.name))


def target_name(number: int, width: int) -> str:
    return f"episode_{number:0{width}d}"


def temp_path(dataset_dir: Path, index: int, source: Path) -> Path:
    return dataset_dir / f"{TEMP_PREFIX}{index:06d}_{source.name}"


def build_plan(
    dataset_dir: Path,
    episodes: list[EpisodeDir],
    start: int,
    width: int,
) -> list[RenameItem]:
    source_paths = {item.path for item in episodes}
    target_paths: list[Path] = []

    for index, item in enumerate(episodes):
        new_number = start + index
        target_paths.append(dataset_dir / target_name(new_number, width))

    duplicate_targets = {
        path for path in target_paths if target_paths.count(path) > 1
    }
    if duplicate_targets:
        names = ", ".join(path.name for path in sorted(duplicate_targets))
        raise ValueError(f"Duplicate target names generated: {names}")

    external_conflicts = [
        path for path in target_paths if path.exists() and path not in source_paths
    ]
    if external_conflicts:
        names = ", ".join(path.name for path in external_conflicts)
        raise FileExistsError(f"Target already exists but is not an episode source: {names}")

    rename_items: list[RenameItem] = []
    for index, (item, target) in enumerate(zip(episodes, target_paths), 1):
        if item.path == target:
            continue
        tmp = temp_path(dataset_dir, index, item.path)
        if tmp.exists():
            raise FileExistsError(f"Temporary path already exists: {tmp}")
        rename_items.append(RenameItem(source=item.path, target=target, temp=tmp))

    return rename_items


def print_plan(
    dataset_dir: Path,
    episodes: list[EpisodeDir],
    rename_items: list[RenameItem],
    start: int,
    width: int,
    preview: int,
    apply: bool,
) -> None:
    end = start + len(episodes) - 1
    mode = "APPLY" if apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Dataset: {dataset_dir}")
    print(f"Episodes found: {len(episodes)}")
    print(
        f"Target sequence: {target_name(start, width)}-{target_name(end, width)}"
        if episodes
        else "Target sequence: <empty>"
    )
    print(f"Need rename: {len(rename_items)}")
    print()

    if not rename_items:
        print("Already sequential. Nothing to rename.")
        return

    print("Rename plan:")
    shown_items = rename_items[:preview]
    for item in shown_items:
        print(f"  {item.source.name} -> {item.target.name}")
    hidden = len(rename_items) - len(shown_items)
    if hidden > 0:
        print(f"  ... {hidden} more")
    print()

    if not apply:
        print("No files changed. Add --apply to rename directories.")


def apply_plan(rename_items: list[RenameItem]) -> None:
    for item in rename_items:
        item.source.rename(item.temp)

    for item in rename_items:
        if item.target.exists():
            raise FileExistsError(f"Target exists before final rename: {item.target}")
        item.temp.rename(item.target)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if args.start < 0:
        raise ValueError(f"--start must be >= 0: {args.start}")
    if args.width < 1:
        raise ValueError(f"--width must be >= 1: {args.width}")

    episodes = iter_episode_dirs(dataset_dir)
    rename_items = build_plan(dataset_dir, episodes, args.start, args.width)
    print_plan(
        dataset_dir=dataset_dir,
        episodes=episodes,
        rename_items=rename_items,
        start=args.start,
        width=args.width,
        preview=args.preview,
        apply=args.apply,
    )

    if args.apply and rename_items:
        apply_plan(rename_items)
        print(f"Renamed {len(rename_items)} episode directories.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
