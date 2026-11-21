from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class SceneEncoder:
    def __init__(self, model_name: str) -> None:
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        images = [Image.fromarray(frame) for frame in frames]
        inputs = self.processor(images=images, return_tensors="pt")
        vectors = self.model(**inputs).image_embeds
        vectors = torch.nn.functional.normalize(vectors, dim=1)
        scene_vector = torch.nn.functional.normalize(vectors.mean(dim=0), dim=0)
        return scene_vector.numpy()


def encode_scenes(
    encoder: SceneEncoder,
    keyframes: Iterable[Sequence[np.ndarray]],
    *,
    executor: ThreadPoolExecutor | None,
    workers: int,
) -> np.ndarray:
    if executor is None:
        return np.stack([encoder.encode(frames) for frames in keyframes])

    # Bound queued frames while preserving the order of the input scenes.
    pending = deque()
    vectors = []
    for frames in keyframes:
        pending.append(executor.submit(encoder.encode, frames))
        if len(pending) >= 2 * workers:
            vectors.append(pending.popleft().result())
    while pending:
        vectors.append(pending.popleft().result())
    return np.stack(vectors)
