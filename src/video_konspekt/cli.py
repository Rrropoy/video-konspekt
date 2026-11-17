import argparse
import math
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="video-konspekt", description="Create a visual summary of a video."
    )
    parser.add_argument("input", type=Path, help="input video file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output MP4 file")
    parser.add_argument("--clips", type=positive_int, default=8, help="maximum number of clips")
    parser.add_argument(
        "--clip-seconds", type=positive_float, default=5.0,
        help="maximum clip duration in seconds",
    )
    parser.add_argument(
        "--scene-threshold", type=positive_float, default=27.0,
        help="scene change threshold",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for k-means")
    parser.add_argument(
        "--model", default="openai/clip-vit-base-patch32",
        help="CLIP model name or local directory",
    )
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.suffix.lower() != ".mp4":
        parser.error("the output file must have an .mp4 extension")
    if source == output:
        parser.error("the input and output files must be different")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    try:
        from .pipeline import summarize

        report, count = summarize(
            source, output, count=args.clips, clip_seconds=args.clip_seconds,
            threshold=args.scene_threshold, seed=args.seed, model=args.model,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Video summary ({count} clips): {output}")
    print(f"Timestamps and timings: {report}")
    return 0
