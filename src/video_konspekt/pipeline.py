import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import torch
from threadpoolctl import threadpool_limits

from .clustering import fit_kmeans, select_representatives
from .embeddings import SceneEncoder, encode_scenes
from .export import Clip, export_summary, find_tools
from .scenes import Scene, detect_scenes, iter_keyframes


def clip_from_scene(scene: Scene, frame_limit: int) -> Clip:
    length = min(scene.end_frame - scene.start_frame, frame_limit)
    start = scene.start_frame + (scene.end_frame - scene.start_frame - length) // 2
    return Clip(start, start + length)


def summarize(
    source: Path,
    output: Path,
    *,
    count: int,
    clip_seconds: float,
    threshold: float,
    seed: int,
    model: str,
    workers: int = 1,
) -> tuple[Path, int]:
    if workers < 1:
        raise ValueError("Workers must be greater than zero.")
    report_path = output.with_suffix(".timings.json")
    ffmpeg, ffprobe = find_tools()
    if not source.is_file():
        raise ValueError(f"Input video not found: {source}")
    for path in (output, report_path):
        if path.exists():
            raise ValueError(f"File already exists: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    timings = {}
    with threadpool_limits(limits=1):
        started = perf_counter()
        start = perf_counter()
        scenes, fps = detect_scenes(source, threshold)
        timings["scene_detection"] = perf_counter() - start

        start = perf_counter()
        encoder = SceneEncoder(model)
        timings["model_loading"] = perf_counter() - start

        active_workers = min(workers, len(scenes))
        pool = (
            ThreadPoolExecutor(
                max_workers=active_workers,
                initializer=getattr(torch, "init_num_threads", None),
            )
            if active_workers > 1
            else nullcontext()
        )
        with pool as executor:
            start = perf_counter()
            with closing(iter_keyframes(source, scenes)) as keyframes:
                vectors = encode_scenes(
                    encoder, keyframes, executor=executor, workers=active_workers
                )
            timings["keyframes_and_embeddings"] = perf_counter() - start

            start = perf_counter()
            centers, labels = fit_kmeans(
                vectors, count, seed, executor=executor, workers=active_workers
            )
            timings["clustering"] = perf_counter() - start

            start = perf_counter()
            selected = select_representatives(vectors, centers, labels, executor=executor)
            frame_limit = max(1, int(clip_seconds * fps))
            clips = [clip_from_scene(scenes[index], frame_limit) for index in selected]
            timings["representative_selection"] = perf_counter() - start

        with TemporaryDirectory(prefix=".video-konspekt-", dir=output.parent) as temp:
            staged_video = Path(temp) / "summary.mp4"
            staged_report = Path(temp) / "report.json"
            start = perf_counter()
            export_summary(source, staged_video, clips, fps, ffmpeg, ffprobe)
            timings["video_export"] = perf_counter() - start
            timings["total"] = perf_counter() - started

            report = {
                "source": str(source),
                "output": str(output),
                "model": model,
                "fps": fps,
                "scene_count": len(scenes),
                "requested_clips": count,
                "clip_seconds": clip_seconds,
                "scene_threshold": threshold,
                "seed": seed,
                "workers": workers,
                "active_workers": active_workers,
                "selected_clips": [
                    {
                        "scene_index": index,
                        "start_frame": clip.start_frame,
                        "end_frame_exclusive": clip.end_frame,
                        "start_seconds": clip.start_frame / fps,
                        "end_seconds": clip.end_frame / fps,
                    }
                    for index, clip in zip(selected, clips, strict=True)
                ],
                "seconds": timings,
            }
            staged_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staged_video.rename(output)
            staged_report.rename(report_path)

    return report_path, len(clips)
