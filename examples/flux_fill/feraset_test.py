from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Config:
    device: str
    model_id: str
    num_inference_steps: int
    width: int
    height: int
    dtype: torch.dtype
    guidance_scale: Optional[float] = None
    max_sequence_length: Optional[int] = None
    strength: Optional[float] = None
    lora_path: Optional[str] = None
    lora_scale: Optional[float] = None
    start_step: Optional[int] = None
    id_weight: Optional[float] = None
    true_cfg: Optional[float] = None
    timestep_to_start_cfg: Optional[int] = None
    gamma: Optional[float] = None
    eta: Optional[float] = None
    s: Optional[float] = None
    tau: Optional[float] = None
    perform_inversion: Optional[bool] = None
    perform_reconstruction: Optional[bool] = None
    perform_editing: Optional[bool] = None
    inversion_true_cfg: Optional[float] = None
    mask_inject_steps: Optional[int] = None
    

import numpy as np
import PIL
from PIL import Image

MIN_ASPECT_RATIO = 9 / 16
MAX_ASPECT_RATIO = 16 / 9
FIXED_DIMENSION = 1024

config = Config(
    model_id="black-forest-labs/FLUX.1-Fill-dev",
    dtype=torch.bfloat16,
    device="cuda",
    guidance_scale=50,
    num_inference_steps=5,
    max_sequence_length=512,
    width=1024,
    height=1536,
)


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image


def calculate_optimal_dimensions(image: Image.Image):
    original_width, original_height = image.size

    original_aspect_ratio = original_width / original_height
    if original_aspect_ratio > 1:
        width = FIXED_DIMENSION
        height = round(FIXED_DIMENSION / original_aspect_ratio)
    else:
        height = FIXED_DIMENSION
        width = round(FIXED_DIMENSION * original_aspect_ratio)

    width = (width // 8) * 8
    height = (height // 8) * 8

    calculated_aspect_ratio = width / height
    if calculated_aspect_ratio > MAX_ASPECT_RATIO:
        width = int((height * MAX_ASPECT_RATIO // 8) * 8)
    elif calculated_aspect_ratio < MIN_ASPECT_RATIO:
        height = int((width / MIN_ASPECT_RATIO // 8) * 8)

    width = int(max(width, 576)) if width == FIXED_DIMENSION else width
    height = int(max(height, 576)) if height == FIXED_DIMENSION else height

    return width, height

import cv2
import numpy as np
import torch
import torch.nn as nn
from diffusers import FluxFillPipeline
from PIL import Image
from scipy.ndimage import binary_dilation
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

#from src.muscle_img2img.utils import calculate_optimal_dimensions, resize_img
#from src.utils.constants import Config
TOKEN = "<huggingface_token>"
SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
PROMPT = "muscular, wider shoulder,bigger arms,sporty looking"
# PROMPT = "a superman costume"
ITERATION = 20


def segformer_seg(reference_image):
    processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
    model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL_ID).to(config.device)
    inputs = processor(images=reference_image, return_tensors="pt")
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=reference_image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    selected_classes = [4, 7, 14, 15]

    binary_mask = np.isin(pred_seg.numpy(), selected_classes).astype(
        np.uint8
    )  # 1: selected, 0: others
    rgb_mask = np.stack([binary_mask * 255] * 3, axis=-1).astype(np.uint8)

    mask_image = Image.fromarray(rgb_mask)
    mask_np = np.array(mask_image.convert("L"))
    mask_np = mask_np > 128
    final_dilated_mask = binary_dilation(
        mask_np, iterations=ITERATION, structure=np.ones((4, 4))
    )
    blurred_mask = cv2.GaussianBlur(
        (final_dilated_mask * 255).astype(np.uint8), (5, 5), 0
    )
    return Image.fromarray(blurred_mask).convert("RGB")



def main():
    @torch.inference_mode()
    def infer_img2img(prompt, image, generator) -> Image:
        resized_image = resize_img(image, max_side=1024)
        mask = segformer_seg(resized_image)
        final_mask = mask.resize(resized_image.size)
        width, height = calculate_optimal_dimensions(resized_image)
        muscle_image = pipeline(
            prompt=prompt,
            image=resized_image,
            mask_image=final_mask,
            guidance_scale=config.guidance_scale,
            width=width,
            height=height,
            num_inference_steps=config.num_inference_steps,
            max_sequence_length=config.max_sequence_length,
            generator=generator,
        ).images[0]
        torch.cuda.empty_cache()

        return muscle_image

    import random
    import requests
    from io import BytesIO


    pipeline = FluxFillPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        token = TOKEN
    )
    pipeline.to(config.device)

    seed = random.randint(0, 2**63 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    input_img_url = "https://i.ibb.co/xKS4F80b/leonardo-dicaprio-1.jpg"
    response = requests.get(input_img_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    result_image = infer_img2img(
                prompt=PROMPT,
                image=image,
                generator=generator)

    result_image.save("img_feraset.png")
    
if __name__ == "__main__":
    main()