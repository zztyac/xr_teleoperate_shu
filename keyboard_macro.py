#!/usr/bin/env python3
"""Record and replay terminal keystrokes for a keyboard-driven collector.

The collector is launched inside a pseudo terminal (PTY).  During recording,
each input byte sequence is forwarded to the collector and stored together
with its relative timestamp.  During replay, the same byte sequences are sent
back to a newly launched collector process.

Examples:
    python3 keyboard_macro.py record task.json -- python3 collect.py
    python3 keyboard_macro.py replay task.json --repeat 20 --interval 2
    python3 keyboard_macro.py replay task.json -- python3 collect.py --camera 0

Press Ctrl-] at any time to stop this wrapper.  Ctrl-] is never forwarded to
the collector and is never written to the macro file.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import errno
import fcntl
import json
import os
import selectors
import shlex
import signal
import struct
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Sequence


STOP_BYTE = b"\x1d"  # Ctrl-]
FORMAT_VERSION = 1


def _command(value: Sequence[str]) -> list[str]:
    command = list(value)
    if command and command[0] == "--":
        command = command[1:]
    return command


def _display_bytes(data: bytes) -> str:
    names = {
        b"\x1b[A": "UP",
        b"\x1b[B": "DOWN",
        b"\x1b[C": "RIGHT",
        b"\x1b[D": "LEFT",
        b"\r": "ENTER",
        b"\n": "ENTER",
        b"\t": "TAB",
        b"\x7f": "BACKSPACE",
        b" ": "SPACE",
        b"\x03": "CTRL-C",
    }
    if data in names:
        return names[data]
    if len(data) == 1 and 1 <= data[0] <= 26:
        return f"CTRL-{chr(data[0] + 64)}"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "0x" + data.hex()
    return text if text.isprintable() else "0x" + data.hex()


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        view = view[written:]


def _copy_terminal_size(src_fd: int, dst_fd: int) -> None:
    try:
        size = fcntl.ioctl(src_fd, termios.TIOCGWINSZ, b"\0" * 8)
        fcntl.ioctl(dst_fd, termios.TIOCSWINSZ, size)
    except OSError:
        pass


def _spawn(command: Sequence[str]) -> tuple[int, int]:
    if not command:
        raise ValueError("没有可执行的采集命令")
    pid, master_fd = os.forkpty()
    if pid == 0:
        try:
            os.execvp(command[0], list(command))
        except OSError as exc:
            print(f"无法启动 {command[0]!r}: {exc}", file=sys.stderr)
            os._exit(127)
    _copy_terminal_size(sys.stdin.fileno(), master_fd)
    return pid, master_fd


def _child_status(pid: int) -> int | None:
    waited_pid, status = os.waitpid(pid, os.WNOHANG)
    if waited_pid == 0:
        return None
    return os.waitstatus_to_exitcode(status)


def _terminate_child(pid: int, master_fd: int) -> int:
    status = _child_status(pid)
    if status is not None:
        return status
    try:
        _write_all(master_fd, b"\x03")
    except OSError:
        pass
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        status = _child_status(pid)
        if status is not None:
            return status
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        status = _child_status(pid)
        if status is not None:
            return status
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _, wait_status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(wait_status)


class RawTerminal:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdin.isatty()
        self.fd = sys.stdin.fileno()
        self.previous: list[Any] | None = None

    def __enter__(self) -> "RawTerminal":
        if self.enabled:
            self.previous = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        return self

    def __exit__(self, *_: object) -> None:
        if self.previous is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)


def _save_macro(path: Path, command: Sequence[str], events: list[dict[str, Any]]) -> None:
    payload = {
        "format": "xr-keyboard-macro",
        "version": FORMAT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "command": list(command),
        "event_count": len(events),
        "duration": round(events[-1]["time"], 6) if events else 0.0,
        "events": events,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_macro(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取按键文件 {path}: {exc}") from exc
    if payload.get("format") != "xr-keyboard-macro" or payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"不支持的按键文件格式或版本: {path}")
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("按键文件缺少 events 列表")
    previous = -1.0
    for index, event in enumerate(events):
        try:
            timestamp = float(event["time"])
            base64.b64decode(event["data"], validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"第 {index + 1} 个按键事件无效") from exc
        if timestamp < 0 or timestamp < previous:
            raise ValueError("按键事件时间必须非负并按升序排列")
        previous = timestamp
    return payload


def record(args: argparse.Namespace) -> int:
    command = _command(args.command)
    if not command:
        raise ValueError("record 后必须使用 -- 指定采集命令")
    if not sys.stdin.isatty():
        raise ValueError("录制必须在交互式终端中运行")

    output = Path(args.output).expanduser()
    if output.exists() and not args.force:
        raise ValueError(f"{output} 已存在；如需覆盖请加 --force")

    pid, master_fd = _spawn(command)
    selector = selectors.DefaultSelector()
    selector.register(sys.stdin.fileno(), selectors.EVENT_READ, "stdin")
    selector.register(master_fd, selectors.EVENT_READ, "child")
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    stopped = False
    exit_code: int | None = None

    print("[keyboard_macro] 已开始录制；按 Ctrl-] 结束并保存。", file=sys.stderr)
    try:
        with RawTerminal():
            while exit_code is None and not stopped:
                _copy_terminal_size(sys.stdin.fileno(), master_fd)
                for key, _ in selector.select(timeout=0.1):
                    if key.data == "stdin":
                        data = os.read(sys.stdin.fileno(), 4096)
                        if not data:
                            stopped = True
                            break
                        stop_at = data.find(STOP_BYTE)
                        forwarded = data if stop_at < 0 else data[:stop_at]
                        if forwarded:
                            timestamp = round(time.monotonic() - started, 6)
                            events.append({
                                "time": timestamp,
                                "data": base64.b64encode(forwarded).decode("ascii"),
                                "key": _display_bytes(forwarded),
                            })
                            _write_all(master_fd, forwarded)
                        if stop_at >= 0:
                            stopped = True
                            break
                    else:
                        try:
                            data = os.read(master_fd, 65536)
                        except OSError as exc:
                            if exc.errno != errno.EIO:
                                raise
                            data = b""
                        if data:
                            _write_all(sys.stdout.fileno(), data)
                exit_code = _child_status(pid)
    finally:
        selector.close()
        if exit_code is None:
            exit_code = _terminate_child(pid, master_fd)
        os.close(master_fd)
        _save_macro(output, command, events)

    print(
        f"\n[keyboard_macro] 已保存 {len(events)} 个事件到 {output} "
        f"（时长 {events[-1]['time'] if events else 0:.3f} 秒）。",
        file=sys.stderr,
    )
    return 0 if stopped else int(exit_code or 0)


def replay(args: argparse.Namespace) -> int:
    if args.repeat < 1:
        raise ValueError("--repeat 必须至少为 1")
    if args.speed <= 0:
        raise ValueError("--speed 必须大于 0")
    if args.interval < 0 or args.start_delay < 0 or args.finish_delay < 0:
        raise ValueError("延时参数不能为负数")

    payload = _load_macro(Path(args.input).expanduser())
    command = _command(args.command) or payload.get("command", [])
    if not command:
        raise ValueError("按键文件中没有命令，请在 replay 后使用 -- 指定采集命令")

    decoded_events = [
        (float(event["time"]) / args.speed, base64.b64decode(event["data"]))
        for event in payload["events"]
    ]
    pid, master_fd = _spawn(command)
    selector = selectors.DefaultSelector()
    selector.register(master_fd, selectors.EVENT_READ, "child")
    if sys.stdin.isatty():
        selector.register(sys.stdin.fileno(), selectors.EVENT_READ, "stdin")

    schedule: list[tuple[float, bytes, int]] = []
    sequence_duration = decoded_events[-1][0] if decoded_events else 0.0
    base = time.monotonic() + args.start_delay
    for repetition in range(args.repeat):
        for timestamp, data in decoded_events:
            schedule.append((base + timestamp, data, repetition + 1))
        base += sequence_duration + args.interval

    next_event = 0
    stopped = False
    exit_code: int | None = None
    finish_at: float | None = None
    print(
        f"[keyboard_macro] 将回放 {len(decoded_events)} 个事件 × {args.repeat} 次，"
        f"速度 {args.speed:g}x；按 Ctrl-] 可紧急停止。",
        file=sys.stderr,
    )
    try:
        with RawTerminal():
            while exit_code is None and not stopped:
                now = time.monotonic()
                while next_event < len(schedule) and schedule[next_event][0] <= now:
                    _, data, repetition = schedule[next_event]
                    _write_all(master_fd, data)
                    next_event += 1
                    if next_event == len(schedule) or schedule[next_event][2] != repetition:
                        print(
                            f"\r\n[keyboard_macro] 第 {repetition}/{args.repeat} 次按键已发送。",
                            file=sys.stderr,
                        )
                if next_event == len(schedule) and finish_at is None:
                    finish_at = now + args.finish_delay
                if args.terminate_after and finish_at is not None and now >= finish_at:
                    stopped = True
                    break

                timeout = 0.1
                if next_event < len(schedule):
                    timeout = max(0.0, min(timeout, schedule[next_event][0] - now))
                for key, _ in selector.select(timeout=timeout):
                    if key.data == "child":
                        try:
                            data = os.read(master_fd, 65536)
                        except OSError as exc:
                            if exc.errno != errno.EIO:
                                raise
                            data = b""
                        if data:
                            _write_all(sys.stdout.fileno(), data)
                    else:
                        data = os.read(sys.stdin.fileno(), 4096)
                        if STOP_BYTE in data:
                            stopped = True
                            break
                        # Before/after automatic playback, allow manual emergency input.
                        if data:
                            _write_all(master_fd, data)
                exit_code = _child_status(pid)
    finally:
        selector.close()
        if exit_code is None:
            exit_code = _terminate_child(pid, master_fd)
        os.close(master_fd)

    if next_event < len(schedule) and not stopped:
        print(
            f"\n[keyboard_macro] 采集程序提前退出（退出码 {exit_code}），"
            f"仅发送 {next_event}/{len(schedule)} 个事件。",
            file=sys.stderr,
        )
        return int(exit_code or 1)
    print("\n[keyboard_macro] 回放结束。", file=sys.stderr)
    return int(exit_code or 0) if not args.terminate_after else 0


def show(args: argparse.Namespace) -> int:
    payload = _load_macro(Path(args.input).expanduser())
    print(f"文件: {args.input}")
    print(f"创建时间: {payload.get('created_at', '未知')}")
    print(f"原命令: {shlex.join(payload.get('command', []))}")
    print(f"事件数: {len(payload['events'])}")
    print(f"时长: {float(payload.get('duration', 0)):.3f} 秒")
    print("\n序号      时间(s)  按键")
    for index, event in enumerate(payload["events"], start=1):
        print(f"{index:4d}  {float(event['time']):10.3f}  {event.get('key', '?')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="录制并自动回放终端键盘控制序列。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s record macros/button.json -- python3 collect.py\n"
            "  %(prog)s show macros/button.json\n"
            "  %(prog)s replay macros/button.json --repeat 10 --interval 2\n"
            "  %(prog)s replay macros/button.json --speed 0.8 -- python3 collect.py"
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    record_parser = subparsers.add_parser("record", help="录制并转发一次人工按键过程")
    record_parser.add_argument("output", help="输出 JSON 文件")
    record_parser.add_argument("--force", action="store_true", help="覆盖已有文件")
    record_parser.add_argument("command", nargs=argparse.REMAINDER, help="-- 后为原采集命令")
    record_parser.set_defaults(handler=record)

    replay_parser = subparsers.add_parser("replay", help="自动回放已录制的按键过程")
    replay_parser.add_argument("input", help="record 生成的 JSON 文件")
    replay_parser.add_argument("--repeat", type=int, default=1, help="回放次数（默认 1）")
    replay_parser.add_argument("--speed", type=float, default=1.0, help="回放速度（默认 1.0）")
    replay_parser.add_argument("--interval", type=float, default=1.0, help="多次回放间隔秒数")
    replay_parser.add_argument("--start-delay", type=float, default=3.0, help="开始前等待秒数")
    replay_parser.add_argument("--finish-delay", type=float, default=1.0, help="最后按键后等待秒数")
    replay_parser.add_argument(
        "--terminate-after",
        action="store_true",
        help="最后一次回放后终止采集程序；默认继续运行并允许手动接管",
    )
    replay_parser.add_argument("command", nargs=argparse.REMAINDER, help="可选：覆盖录制时的采集命令")
    replay_parser.set_defaults(handler=replay)

    show_parser = subparsers.add_parser("show", help="查看录制结果")
    show_parser.add_argument("input", help="record 生成的 JSON 文件")
    show_parser.set_defaults(handler=show)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
