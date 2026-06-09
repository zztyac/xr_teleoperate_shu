#!/usr/bin/env python3
"""Fix the text metadata for early vehicle physical button episodes.

Default target:
    /mnt/data/zty/json_data/vehicle_physical_button_press_ccw/episode_0001
    through episode_0056

Dry run:
    python3 scripts/fix_vehicle_button_text.py --dry-run

Apply:
    python3 scripts/fix_vehicle_button_text.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path(
    "/mnt/data/zty/json_data/vehicle_physical_button_press_ccw"
)
DEFAULT_MAX_EPISODE = 56

SHORT_PRESS_STEPS = (
    "step1: move the dexterous hand to the target button; "
    "step2: align the fingertip with the button surface; "
    "step3: press the button once; "
    "step4: release the button; "
    "step5: return to a safe pose;"
)

TARGET_TEXT = {
    "goal": "Press the front windshield defrost button once.",
    "desc": (
        "Collect a demonstration for the in-car physical button task: "
        "front windshield defrost."
    ),
    "steps": SHORT_PRESS_STEPS,
}

EPISODE_RE = re.compile(r"episode_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace data.json -> text for vehicle_physical_button_press_ccw "
            "episodes up to a given episode number."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--max-episode",
        type=int,
        default=DEFAULT_MAX_EPISODE,
        help=f"Replace episodes with number <= this value. Default: {DEFAULT_MAX_EPISODE}",
    )
    parser.add_argument(
        "--min-episode",
        type=int,
        default=0,
        help="Replace episodes with number >= this value. Default: 0",
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


def episode_number(path: Path) -> int | None:
    match = EPISODE_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1))


def iter_episode_data_files(
    dataset_dir: Path,
    min_episode: int,
    max_episode: int,
) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for episode_dir in dataset_dir.iterdir():
        if not episode_dir.is_dir():
            continue
        number = episode_number(episode_dir)
        if number is None or number < min_episode or number > max_episode:
            continue
        data_path = episode_dir / "data.json"
        if data_path.exists():
            files.append((number, data_path))
        else:
            print(f"SKIP missing data.json: {episode_dir}")
    return sorted(files)


def skip_ws(raw: str, idx: int) -> int:
    while idx < len(raw) and raw[idx] in " \t\r\n":
        idx += 1
    return idx


def find_top_level_text_range(raw: str) -> tuple[int, int]:
    """Return the [start, end) character range of the top-level text field."""

    decoder = json.JSONDecoder()
    idx = skip_ws(raw, 0)
    if idx >= len(raw) or raw[idx] != "{":
        raise ValueError("JSON root is not an object")

    idx += 1
    while True:
        idx = skip_ws(raw, idx)
        if idx >= len(raw):
            raise ValueError("Unterminated JSON object")
        if raw[idx] == "}":
            break
        if raw[idx] != '"':
            raise ValueError(f"Expected object key at character {idx}")

        key_start = idx
        key, key_end = decoder.raw_decode(raw, idx)
        if not isinstance(key, str):
            raise ValueError(f"Expected string object key at character {idx}")

        idx = skip_ws(raw, key_end)
        if idx >= len(raw) or raw[idx] != ":":
            raise ValueError(f"Expected ':' after key {key!r}")

        value_start = skip_ws(raw, idx + 1)
        _, value_end = decoder.raw_decode(raw, value_start)

        if key == "text":
            return key_start, value_end

        idx = skip_ws(raw, value_end)
        if idx < len(raw) and raw[idx] == ",":
            idx += 1
            continue
        if idx < len(raw) and raw[idx] == "}":
            break
        raise ValueError(f"Expected ',' or '}}' after key {key!r}")

    raise ValueError("Top-level 'text' field not found")


def make_text_field(raw: str) -> str:
    newline = "\r\n" if "\r\n" in raw else "\n"
    text_json = json.dumps(TARGET_TEXT, ensure_ascii=False, indent=4)
    return '"text": ' + text_json.replace("\n", newline)


def replace_text_field(raw: str) -> tuple[str, bool]:
    parsed: dict[str, Any] = json.loads(raw)
    if parsed.get("text") == TARGET_TEXT:
        return raw, False

    start, end = find_top_level_text_range(raw)
    updated = raw[:start] + make_text_field(raw) + raw[end:]

    updated_parsed: dict[str, Any] = json.loads(updated)
    if updated_parsed.get("text") != TARGET_TEXT:
        raise ValueError("Updated JSON did not contain the expected text metadata")

    return updated, True


def write_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    files = iter_episode_data_files(dataset_dir, args.min_episode, args.max_episode)
    if not files:
        print("No matching episode data.json files found.")
        return 0

    changed = 0
    unchanged = 0
    for number, data_path in files:
        raw = data_path.read_text(encoding="utf-8")
        updated, needs_write = replace_text_field(raw)
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
        write_atomic(data_path, updated)
        print(f"UPDATED episode_{number:04d}: {data_path}")

    mode = "would update" if args.dry_run else "updated"
    print(f"Done: {changed} {mode}, {unchanged} unchanged, {len(files)} checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
