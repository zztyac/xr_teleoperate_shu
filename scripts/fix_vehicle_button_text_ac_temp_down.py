#!/usr/bin/env python3
"""Fix the text metadata for a selected vehicle physical button episode range.

Dry run:
    python3 scripts/fix_vehicle_button_text_ac_temp_down.py --min-episode 58 --max-episode 61 --dry-run

Apply:
    python3 scripts/fix_vehicle_button_text_ac_temp_down.py --min-episode 58 --max-episode 61
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fix_vehicle_button_text as fixer


DEFAULT_DATASET_DIR = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press_ccw"
)

TARGET_TEXT = {
    "goal": "Press the air conditioning temperature down button once.",
    "desc": (
        "Collect a demonstration for the in-car physical button task: "
        "air conditioning temperature down."
    ),
    "steps": fixer.SHORT_PRESS_STEPS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace data.json -> text for vehicle_physical_button_press_ccw "
            "episodes in the specified range."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--min-episode",
        "--start",
        dest="min_episode",
        type=int,
        required=True,
        help="Replace episodes with number >= this value.",
    )
    parser.add_argument(
        "--max-episode",
        "--end",
        dest="max_episode",
        type=int,
        required=True,
        help="Replace episodes with number <= this value.",
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


def print_target_text() -> None:
    print("Current replacement task language:")
    print(f"  goal: {TARGET_TEXT['goal']}")
    print(f"  desc: {TARGET_TEXT['desc']}")
    print(f"  steps: {TARGET_TEXT['steps']}")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if args.min_episode > args.max_episode:
        raise ValueError(
            f"min episode must be <= max episode: "
            f"{args.min_episode} > {args.max_episode}"
        )

    fixer.TARGET_TEXT = TARGET_TEXT
    print_target_text()
    print(f"Episode range: episode_{args.min_episode:04d}-episode_{args.max_episode:04d}")
    print()

    files = fixer.iter_episode_data_files(
        dataset_dir,
        args.min_episode,
        args.max_episode,
    )
    if not files:
        print("No matching episode data.json files found.")
        return 0

    changed = 0
    unchanged = 0
    for number, data_path in files:
        raw = data_path.read_text(encoding="utf-8")
        updated, needs_write = fixer.replace_text_field(raw)
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
    print(f"Done: {changed} {mode}, {unchanged} unchanged, {len(files)} checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
