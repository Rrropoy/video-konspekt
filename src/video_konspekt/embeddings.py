from collections.abc import Sequence

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
