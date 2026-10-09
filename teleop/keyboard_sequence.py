#!/usr/bin/env python3
"""Record and replay timed keyboard sequences for repetitive data collection.

Requires: pip install pynput
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from pynput import keyboard
except ImportError:
    print("Missing dependency: pynput (install it with: pip install pynput)", file=sys.stderr)
    raise SystemExit(2)


FORMAT_VERSION = 1


def encode_key(key: keyboard.Key | keyboard.KeyCode) -> dict[str, Any]:
    if isinstance(key, keyboard.KeyCode):
        return {"kind": "char", "value": key.char}
    return {"kind": "special", "value": key.name}


def decode_key(data: dict[str, Any]) -> keyboard.Key | keyboard.KeyCode:
    if data["kind"] == "char":
        return keyboard.KeyCode.from_char(data["value"])
    try:
        return getattr(keyboard.Key, data["value"])
    except AttributeError as exc:
        raise ValueError(f"Unknown special key: {data['value']}") from exc


def save_sequence(path: Path, events: list[dict[str, Any]]) -> None:
    payload = {
        "version": FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "events": events,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_sequence(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != FORMAT_VERSION or not isinstance(payload.get("events"), list):
        raise ValueError(f"Unsupported or invalid sequence file: {path}")
    return payload["events"]


def record(path: Path, start_key: keyboard.Key, stop_key: keyboard.Key) -> None:
    events: list[dict[str, Any]] = []
    recording = False
    started_at = 0.0
    pressed: set[str] = set()

    print(f"Press {start_key.name} to start recording; press {stop_key.name} to finish and save.")

    def key_id(key: keyboard.Key | keyboard.KeyCode) -> str:
        return json.dumps(encode_key(key), ensure_ascii=False, sort_keys=True)

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        nonlocal recording, started_at
        if key == stop_key:
            return False
        if not recording:
            if key == start_key:
                recording = True
                started_at = time.monotonic()
                print("Recording... perform the task now.")
            return None

        ident = key_id(key)
        # Ignore OS key-repeat; one physical press should produce one press event.
        if ident not in pressed:
            pressed.add(ident)
            events.append({"time": time.monotonic() - started_at, "action": "press", "key": encode_key(key)})
        return None

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        if not recording or key in (start_key, stop_key):
            return
        pressed.discard(key_id(key))
        events.append({"time": time.monotonic() - started_at, "action": "release", "key": encode_key(key)})

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    if not recording:
        raise SystemExit("Recording cancelled before it started.")
    save_sequence(path, events)
    duration = events[-1]["time"] if events else 0.0
    print(f"Saved {len(events)} events ({duration:.3f}s) to {path}")


def replay(path: Path, repeat: int, speed: float, delay: float, interval: float) -> None:
    events = load_sequence(path)
    if not events:
        raise SystemExit("Sequence contains no keyboard events.")

    stop = threading.Event()

    def emergency_stop(key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        if key == keyboard.Key.esc:
            stop.set()
            return False
        return None

    listener = keyboard.Listener(on_press=emergency_stop)
    listener.start()
    controller = keyboard.Controller()

    print(f"Replay starts in {delay:g}s. Focus the collection window. Press Esc at any time to stop.")
    if stop.wait(delay):
        return

    try:
        for run_index in range(repeat):
            if stop.is_set():
                break
            start = time.monotonic()
            print(f"Run {run_index + 1}/{repeat}")
            for event in events:
                deadline = start + float(event["time"]) / speed
                if stop.wait(max(0.0, deadline - time.monotonic())):
                    break
                key = decode_key(event["key"])
                if event["action"] == "press":
                    controller.press(key)
                elif event["action"] == "release":
                    controller.release(key)
                else:
                    raise ValueError(f"Unknown action: {event['action']}")
            if run_index + 1 < repeat and stop.wait(interval):
                break
    finally:
        # Ensure no recorded key remains held after errors or emergency stop.
        for event in events:
            try:
                controller.release(decode_key(event["key"]))
            except Exception:
                pass
        listener.stop()
    print("Replay stopped." if stop.is_set() else "Replay completed.")


def parse_special_key(name: str) -> keyboard.Key:
    try:
        return getattr(keyboard.Key, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"Unknown special key: {name}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    recorder = subparsers.add_parser("record", help="record a timed keyboard sequence")
    recorder.add_argument("file", type=Path, help="output JSON file")
    recorder.add_argument("--start-key", type=parse_special_key, default=keyboard.Key.f8)
    recorder.add_argument("--stop-key", type=parse_special_key, default=keyboard.Key.f9)

    player = subparsers.add_parser("replay", help="replay a recorded sequence")
    player.add_argument("file", type=Path, help="input JSON file")
    player.add_argument("--repeat", type=int, default=1, help="number of repetitions")
    player.add_argument("--speed", type=float, default=1.0, help="timing multiplier; 2.0 is twice as fast")
    player.add_argument("--delay", type=float, default=5.0, help="seconds before first replay")
    player.add_argument("--interval", type=float, default=2.0, help="seconds between repetitions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "record":
        record(args.file, args.start_key, args.stop_key)
    else:
        if args.repeat < 1 or args.speed <= 0 or args.delay < 0 or args.interval < 0:
            raise SystemExit("repeat must be >= 1, speed > 0, and delays >= 0")
        replay(args.file, args.repeat, args.speed, args.delay, args.interval)


if __name__ == "__main__":
    main()
