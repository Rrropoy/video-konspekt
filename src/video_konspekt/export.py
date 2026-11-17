import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Clip:
    start_frame: int
    end_frame: int


def find_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("Install FFmpeg and add ffmpeg and ffprobe to PATH.")
    return ffmpeg, ffprobe


def run_tool(command: list[str]) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode:
        details = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"{Path(command[0]).stem}: {details}")
    return result.stdout


def export_summary(
    source: Path,
    target: Path,
    clips: list[Clip],
    fps: float,
    ffmpeg: str,
    ffprobe: str,
) -> None:
    audio_streams = run_tool([
        ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(source),
    ])
    has_audio = bool(audio_streams.strip())
    filters = []
    concat_inputs = []
    if has_audio:
        audio_filter = "[0:a:0]aresample=async=1:first_pts=0,apad"
        if len(clips) > 1:
            audio_filter += f",asplit={len(clips)}"
        filters.append(audio_filter + "".join(f"[audio{i}]" for i in range(len(clips))))

    for i, clip in enumerate(clips):
        filters.append(
            f"[0:v:0]trim=start_frame={clip.start_frame}:end_frame={clip.end_frame},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        concat_inputs.append(f"[v{i}]")
        if has_audio:
            start = clip.start_frame / fps
            end = clip.end_frame / fps
            filters.append(
                f"[audio{i}]atrim=start={start:.9f}:end={end:.9f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )
            concat_inputs.append(f"[a{i}]")

    outputs = "[joined][aout]" if has_audio else "[joined]"
    filters.append(
        f"{''.join(concat_inputs)}concat=n={len(clips)}:v=1:a={int(has_audio)}{outputs}"
    )
    filters.append("[joined]pad=ceil(iw/2)*2:ceil(ih/2)*2[vout]")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-threads", "1", "-i", str(source),
        "-filter_complex_threads", "1", "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-c:v", "libx264", "-threads:v", "1",
        "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "20", "-movflags", "+faststart",
    ]
    if has_audio:
        command.extend(["-map", "[aout]", "-c:a", "aac"])
    command.append(str(target))
    run_tool(command)
