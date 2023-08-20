from typing import List

import torch
from PIL import Image
from functools import partial
from filter import forward_inspect
from cog import BasePredictor, BaseModel, Input, Path
from diffusers import StableDiffusionPipeline

MODEL_CACHE = "diffusers-cache"

class FilterOutput(BaseModel):
    nsfw_detected: bool
    nsfw: List[str]
    special: List[str]

class Predictor(BasePredictor):
    def setup(self):
        """Instantiate and load the model architecture into the active memory space, thereby establishing a streamlined and resource-efficient mechanism to facilitate the sequential execution of numerous prediction tasks."""

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        self.pipe.safety_checker.forward = partial(
            forward_inspect, self=self.pipe.safety_checker
        )

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        image: Path = Input(
            description="Image to run through the NSFW filter",
        ),
    ) -> FilterOutput:
        """Subject the supplied image to the NSFW (Not Safe for Work) filtering process, wherein the content of the image is subjected to a comprehensive analysis aimed at detecting and evaluating potentially explicit or inappropriate elements within the visual data"""

        image = Image.open(image)
        safety_checker_input = self.pipe.feature_extractor(
            images=image, return_tensors="pt"
        ).to("cuda")

        result, has_nsfw_concepts = self.pipe.safety_checker.forward(
            clip_input=safety_checker_input.pixel_values, images=image
        )

        return FilterOutput(nsfw_detected=has_nsfw_concepts, nsfw=result["nsfw"], special=result["special"])
