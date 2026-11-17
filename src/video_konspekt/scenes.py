from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video
from scenedetect.video_stream import VideoOpenFailure


@dataclass(frozen=True)
class Scene:
    start_frame: int
    end_frame: int

    def sample_frames(self) -> tuple[int, ...]:
        length = self.end_frame - self.start_frame
        count = min(3, length)
        return tuple(
            self.start_frame + ((2 * i + 1) * length) // (2 * count)
            for i in range(count)
        )


def detect_scenes(path: Path, threshold: float) -> tuple[list[Scene], float]:
    try:
        source = open_video(str(path), backend="opencv")
    except VideoOpenFailure as error:
        raise ValueError(f"Could not open video: {path}") from error
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    try:
        manager.detect_scenes(video=source)
        scenes = [
            Scene(start.get_frames(), end.get_frames())
            for start, end in manager.get_scene_list(start_in_scene=True)
            if end > start
        ]
        if not scenes:
            raise ValueError("The video contains no frames.")
        return scenes, source.frame_rate
    finally:
        source.capture.release()


def iter_keyframes(
    path: Path, scenes: Sequence[Scene]
) -> Iterator[list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        frame_index = 0
        for scene in scenes:
            frames = []
            for target in scene.sample_frames():
                while frame_index < target:
                    if not capture.grab():
                        raise ValueError(f"The video ended before frame {target}.")
                    frame_index += 1
                success, frame = capture.read()
                if not success:
                    raise ValueError(f"Could not read frame {target}.")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_index += 1
            yield frames
    finally:
        capture.release()
