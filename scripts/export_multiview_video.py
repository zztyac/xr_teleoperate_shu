#!/usr/bin/env python3
"""Export multi-view image sequences from an episode directory to tiled video.

Expected default layout:

    episode_0001/
      data.json
      colors/
        000000_color_0.jpg
        000000_color_1.jpg
        ...

The script uses only the Python standard library and ffmpeg.



python3 scripts/export_multiview_video.py \
    /mnt/data/zty/json_data/vehicle_physical_button_press/episode_0100 \
    --output /home/ubuntu/zty/xr_teleoperate_shu/episode_0041_colors_multiview.mp4 \
    --overwrite \
    --ffmpeg /home/ubuntu/zty/openpi/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_FPS = 30.0
DEFAULT_IMAGE_SUBDIR = "colors"
DEFAULT_OUTPUT_SUFFIX = "multiview"
IMAGE_RE = re.compile(
    r"^(?P<frame>\d+)_(?P<stream>[A-Za-z0-9]+)_(?P<view>\d+)"
    r"\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


class ExportError(RuntimeError):
    """Raised for user-fixable export problems."""


@dataclass(frozen=True)
class EpisodePlan:
    input_path: Path
    dataset_dir: Path
    image_dir: Path
    output_path: Path
    stream: str
    fps: float
    views: tuple[int, ...]
    frames: tuple[int, ...]
    images_by_view: dict[int, dict[int, Path]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an episode's multi-view image frames to a tiled MP4. "
            "By default it reads episode/colors/*_color_<view>.jpg."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Episode directories, or image directories containing numbered view frames.",
    )
    parser.add_argument(
        "--image-subdir",
        default=DEFAULT_IMAGE_SUBDIR,
        help="Image subdirectory to use when an input is an episode directory.",
    )
    parser.add_argument(
        "--stream",
        help=(
            "Frame stream name in filenames, for example 'color' for "
            "000000_color_0.jpg. Defaults to the most common stream found."
        ),
    )
    parser.add_argument(
        "--views",
        help="Comma-separated view ids to include, for example 0,1,2,3. Defaults to all detected views.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Number of columns in the output grid. Defaults to ceil(sqrt(number_of_views)).",
    )
    parser.add_argument("--fps", type=float, help="Output FPS. Defaults to data.json metadata, then 30.")
    parser.add_argument("--start", type=int, help="First source frame index to export.")
    parser.add_argument("--end", type=int, help="Last source frame index to export, inclusive.")
    parser.add_argument("--limit", type=int, help="Limit the number of exported frames after slicing.")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Use only frames present in every selected view instead of failing on gaps.",
    )
    parser.add_argument(
        "--cell-size",
        help=(
            "Optional WxH cell size, for example 640x480. Each view is scaled "
            "to fit and padded before tiling."
        ),
    )
    parser.add_argument("--output", type=Path, help="Output file path. Only valid with one input.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for one or more inputs.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable.")
    parser.add_argument("--codec", default="libx264", help="Video codec passed to ffmpeg.")
    parser.add_argument("--preset", default="fast", help="ffmpeg encoder preset.")
    parser.add_argument("--crf", type=int, default=18, help="CRF quality value for libx264/libx265.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep staged image sequences after export.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned ffmpeg command without running it.")
    return parser.parse_args()


def parse_views(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None

    views: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            views.append(int(item))
        except ValueError as exc:
            raise ExportError(f"Invalid view id '{item}' in --views") from exc

    if not views:
        raise ExportError("--views did not contain any view ids")
    return tuple(dict.fromkeys(views))


def parse_cell_size(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None

    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", raw, re.IGNORECASE)
    if not match:
        raise ExportError("--cell-size must look like 640x480")

    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ExportError("--cell-size dimensions must be positive")
    return width, height


def resolve_input(input_path: Path, image_subdir: str) -> tuple[Path, Path]:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise ExportError(f"Input does not exist: {input_path}")
    if not input_path.is_dir():
        raise ExportError(f"Input must be a directory: {input_path}")

    nested_image_dir = input_path / image_subdir
    if nested_image_dir.is_dir():
        return input_path, nested_image_dir
    return input_path.parent, input_path


def load_fps(dataset_dir: Path, image_dir: Path, image_subdir: str) -> float:
    for candidate in (dataset_dir / "data.json", image_dir.parent / "data.json"):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        info = payload.get("info", {})
        section_name = "depth" if "depth" in image_subdir.lower() else "image"
        section = info.get(section_name, {})
        fps = section.get("fps")
        if isinstance(fps, (int, float)) and fps > 0:
            return float(fps)

    return DEFAULT_FPS


def scan_images(image_dir: Path) -> dict[str, dict[int, dict[int, Path]]]:
    by_stream: dict[str, dict[int, dict[int, Path]]] = {}

    for path in image_dir.iterdir():
        if not path.is_file():
            continue

        match = IMAGE_RE.match(path.name)
        if not match:
            continue

        frame = int(match.group("frame"))
        stream = match.group("stream").lower()
        view = int(match.group("view"))
        by_view = by_stream.setdefault(stream, {})
        by_frame = by_view.setdefault(view, {})
        if frame in by_frame:
            raise ExportError(
                f"Duplicate frame/view for stream '{stream}': frame {frame}, view {view}"
            )
        by_frame[frame] = path

    if not by_stream:
        raise ExportError(
            f"No images matching '<frame>_<stream>_<view>.jpg' were found in {image_dir}"
        )

    return by_stream


def select_stream(
    by_stream: dict[str, dict[int, dict[int, Path]]],
    requested_stream: str | None,
) -> str:
    if requested_stream:
        stream = requested_stream.lower()
        if stream not in by_stream:
            available = ", ".join(sorted(by_stream))
            raise ExportError(f"Stream '{requested_stream}' not found. Available streams: {available}")
        return stream

    return max(
        sorted(by_stream),
        key=lambda item: sum(len(frames) for frames in by_stream[item].values()),
    )


def select_frames(
    images_by_view: dict[int, dict[int, Path]],
    views: tuple[int, ...],
    start: int | None,
    end: int | None,
    limit: int | None,
    skip_missing: bool,
) -> tuple[int, ...]:
    frame_sets = [set(images_by_view[view]) for view in views]
    if not all(frame_sets):
        raise ExportError("At least one selected view has no frames")

    default_start = max(min(frames) for frames in frame_sets)
    default_end = min(max(frames) for frames in frame_sets)
    start = default_start if start is None else start
    end = default_end if end is None else end
    if start > end:
        raise ExportError(f"Invalid frame range: start {start} is after end {end}")

    if skip_missing:
        common = set.intersection(*frame_sets)
        frames = [frame for frame in sorted(common) if start <= frame <= end]
    else:
        expected = list(range(start, end + 1))
        missing: list[str] = []
        for view in views:
            available = images_by_view[view]
            for frame in expected:
                if frame not in available:
                    missing.append(f"view {view} frame {frame}")
                    if len(missing) >= 12:
                        break
            if len(missing) >= 12:
                break
        if missing:
            examples = "; ".join(missing)
            raise ExportError(
                "Missing frames in strict mode. "
                f"Examples: {examples}. Use --skip-missing to export the common frames only."
            )
        frames = expected

    if limit is not None:
        if limit <= 0:
            raise ExportError("--limit must be positive")
        frames = frames[:limit]

    if not frames:
        raise ExportError("No frames selected for export")
    return tuple(frames)


def build_output_path(
    input_path: Path,
    output: Path | None,
    out_dir: Path | None,
    image_subdir: str,
    multi_input: bool,
) -> Path:
    if output and multi_input:
        raise ExportError("--output can only be used with a single input")

    if output:
        return output.resolve()

    filename = f"{input_path.name}_{image_subdir}_{DEFAULT_OUTPUT_SUFFIX}.mp4"
    if out_dir:
        return (out_dir / filename).resolve()
    return (input_path / filename).resolve()


def make_plan(input_path: Path, args: argparse.Namespace) -> EpisodePlan:
    dataset_dir, image_dir = resolve_input(input_path, args.image_subdir)
    by_stream = scan_images(image_dir)
    stream = select_stream(by_stream, args.stream)
    images_by_view = by_stream[stream]

    requested_views = parse_views(args.views)
    if requested_views is None:
        views = tuple(sorted(images_by_view))
    else:
        missing_views = [view for view in requested_views if view not in images_by_view]
        if missing_views:
            available = ", ".join(str(view) for view in sorted(images_by_view))
            raise ExportError(f"Views not found: {missing_views}. Available views: {available}")
        views = requested_views

    frames = select_frames(
        images_by_view=images_by_view,
        views=views,
        start=args.start,
        end=args.end,
        limit=args.limit,
        skip_missing=args.skip_missing,
    )
    fps = args.fps if args.fps is not None else load_fps(dataset_dir, image_dir, args.image_subdir)
    if fps <= 0:
        raise ExportError("--fps must be positive")

    return EpisodePlan(
        input_path=input_path.resolve(),
        dataset_dir=dataset_dir,
        image_dir=image_dir,
        output_path=build_output_path(
            input_path=input_path,
            output=args.output,
            out_dir=args.out_dir,
            image_subdir=args.image_subdir,
            multi_input=len(args.inputs) > 1,
        ),
        stream=stream,
        fps=fps,
        views=views,
        frames=frames,
        images_by_view=images_by_view,
    )


def common_suffix(paths: Iterable[Path]) -> str:
    suffixes = {path.suffix.lower() for path in paths}
    if len(suffixes) != 1:
        raise ExportError(f"Mixed image extensions are not supported in one view: {sorted(suffixes)}")
    return next(iter(suffixes))


def link_or_copy(source: Path, dest: Path) -> None:
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def stage_sequences(plan: EpisodePlan, keep_temp: bool) -> tuple[Path, list[str]]:
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f"{plan.input_path.name}_multiview_",
            dir=str(plan.output_path.parent),
        )
    )

    input_patterns: list[str] = []
    try:
        for view in plan.views:
            view_dir = temp_root / f"view_{view}"
            view_dir.mkdir()
            sources = [plan.images_by_view[view][frame] for frame in plan.frames]
            suffix = common_suffix(sources)
            for output_idx, source in enumerate(sources):
                link_or_copy(source, view_dir / f"{output_idx:06d}{suffix}")
            input_patterns.append(str(view_dir / f"%06d{suffix}"))
    except Exception:
        if not keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return temp_root, input_patterns


def cleanup_temp(temp_root: Path, keep_temp: bool) -> None:
    if keep_temp:
        print(f"Kept staged images: {temp_root}")
        return
    shutil.rmtree(temp_root, ignore_errors=True)


def format_fps(fps: float) -> str:
    if fps.is_integer():
        return str(int(fps))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def build_layout(num_inputs: int, cols: int, cell_size: tuple[int, int] | None) -> str:
    positions: list[str] = []
    for idx in range(num_inputs):
        row, col = divmod(idx, cols)
        if cell_size:
            cell_width, cell_height = cell_size
            positions.append(f"{col * cell_width}_{row * cell_height}")
            continue

        row_start = row * cols
        x = "0" if col == 0 else "+".join(f"w{item}" for item in range(row_start, row_start + col))
        y = "0" if row == 0 else "+".join(f"h{item * cols}" for item in range(row))
        positions.append(f"{x}_{y}")

    return "|".join(positions)


def build_filter_graph(num_inputs: int, cols: int, cell_size: tuple[int, int] | None) -> str:
    chains: list[str] = []
    labels: list[str] = []

    for idx in range(num_inputs):
        label = f"v{idx}"
        labels.append(f"[{label}]")
        if cell_size:
            width, height = cell_size
            chains.append(
                f"[{idx}:v]"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1[{label}]"
            )
        else:
            chains.append(f"[{idx}:v]setsar=1[{label}]")

    layout = build_layout(num_inputs=num_inputs, cols=cols, cell_size=cell_size)
    chains.append(
        f"{''.join(labels)}xstack=inputs={num_inputs}:layout={layout}:shortest=1[out]"
    )
    return ";".join(chains)


def build_ffmpeg_command(
    plan: EpisodePlan,
    input_patterns: list[str],
    args: argparse.Namespace,
    cell_size: tuple[int, int] | None,
) -> list[str]:
    cols = args.cols or math.ceil(math.sqrt(len(plan.views)))
    if cols <= 0:
        raise ExportError("--cols must be positive")
    if cols > len(plan.views):
        cols = len(plan.views)

    command = [
        args.ffmpeg,
        "-y" if args.overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    for pattern in input_patterns:
        command.extend(["-framerate", format_fps(plan.fps), "-i", pattern])

    command.extend(
        [
            "-filter_complex",
            build_filter_graph(
                num_inputs=len(plan.views),
                cols=cols,
                cell_size=cell_size,
            ),
            "-map",
            "[out]",
            "-frames:v",
            str(len(plan.frames)),
            "-r",
            format_fps(plan.fps),
            "-c:v",
            args.codec,
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            str(plan.output_path),
        ]
    )
    return command


def print_plan(plan: EpisodePlan) -> None:
    print(
        "Exporting "
        f"{plan.input_path.name}: stream={plan.stream}, "
        f"views={','.join(str(view) for view in plan.views)}, "
        f"frames={plan.frames[0]}..{plan.frames[-1]} ({len(plan.frames)} frames), "
        f"fps={format_fps(plan.fps)}, output={plan.output_path}"
    )


def export_plan(plan: EpisodePlan, args: argparse.Namespace, cell_size: tuple[int, int] | None) -> None:
    print_plan(plan)

    if args.dry_run:
        patterns = [f"<staged>/view_{view}/%06d.jpg" for view in plan.views]
        command = build_ffmpeg_command(plan, patterns, args, cell_size)
        print(subprocess.list2cmdline(command))
        return

    if plan.output_path.exists() and not args.overwrite:
        raise ExportError(f"Output already exists, use --overwrite to replace it: {plan.output_path}")

    temp_root, patterns = stage_sequences(plan, keep_temp=args.keep_temp)
    try:
        command = build_ffmpeg_command(plan, patterns, args, cell_size)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        printable = subprocess.list2cmdline(command)
        raise ExportError(f"ffmpeg failed with exit code {exc.returncode}: {printable}") from exc
    finally:
        cleanup_temp(temp_root, keep_temp=args.keep_temp)


def main() -> int:
    args = parse_args()
    try:
        if args.output and args.out_dir:
            raise ExportError("--output and --out-dir cannot be used together")

        cell_size = parse_cell_size(args.cell_size)
        plans = [make_plan(input_path, args) for input_path in args.inputs]
        for plan in plans:
            export_plan(plan, args, cell_size)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())