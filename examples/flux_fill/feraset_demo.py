from dataclasses import dataclass
from typing import Optional

import cv2
import time
import random
import requests
import numpy as np
from io import BytesIO
from scipy.ndimage import binary_dilation

import PIL
from PIL import Image
from transformers import SegformerImageProcessor

import mithril as ml
from util import load_seg_model
from flux_pipeline import create_pipeline

MIN_ASPECT_RATIO = 9 / 16
MAX_ASPECT_RATIO = 16 / 9
FIXED_DIMENSION = 1024
ITERATION = 20

@dataclass
class Config:
    device: str
    model_id: str
    num_inference_steps: int
    dtype: type
    width: int | None = None
    height: int | None = None
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

config = Config(
    model_id="black-forest-labs/FLUX.1-Fill-dev",
    dtype=ml.bfloat16,
    device="tpu",  # "cuda"
    guidance_scale=50,
    num_inference_steps=20,
    max_sequence_length=512
)

# Create backend for mithril model.
seg_backend = ml.JaxBackend(device=config.device, device_mesh=(4,))
backend = ml.JaxBackend(device=config.device, dtype=config.dtype, device_mesh=(4,))
# seg_backend = ml.TorchBackend(device=config.device)
# backend = ml.TorchBackend(device=config.device, dtype=config.dtype)

SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)

# Load weights and get compiled mithril model for segformer.
segformer_pm, params = load_seg_model(SEG_MODEL_ID, seg_backend)

# Prepare data for Mithril model.
m_key = "decode_head_batch_norm_running_mean"
v_key = "decode_head_batch_norm_running_var"
data = {m_key: params.pop(m_key), v_key: params.pop(v_key)}

def segformer_seg(reference_image):
    inputs = processor(images=reference_image, return_tensors='np')
    data["input"] = seg_backend.array(inputs["pixel_values"])
    data["img_size"] = reference_image.size[::-1]
    # Run segmentation model.
    upsampled_logits = segformer_pm.evaluate(params, data)["upsampled_logits"]
    upsampled_logits = seg_backend.to_device(upsampled_logits, "cpu")
    pred_seg = np.array(seg_backend.argmax(upsampled_logits, axis=1)[0])

    selected_classes = [4, 7, 14, 15]
    # 1: selected, 0: others
    binary_mask = np.isin(pred_seg, selected_classes).astype(np.uint8)
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

input_img_url = "https://i.ibb.co/xKS4F80b/leonardo-dicaprio-1.jpg"
response = requests.get(input_img_url)
image = Image.open(BytesIO(response.content)).convert("RGB")
# path = "examples/flux_fill/berkay.png"
# image = Image.open(path).convert("RGB")

resized_image = resize_img(image, max_side=1024)
width, height = calculate_optimal_dimensions(resized_image)
pipeline = create_pipeline(
    backend=backend,
    guidance=config.guidance_scale,
    width=width,
    height=height,
    num_steps=config.num_inference_steps,
    max_seq_len=512,
    seed=random.randint(0, 2**63 - 1),
)

def infer_img2img(prompt, image) -> Image:
    mask = segformer_seg(image)
    final_mask = mask.resize(image.size)
    return pipeline(prompt=prompt, img_cond=image, img_mask=final_mask)

prompts = [
    "muscular wider shoulder, bigger arms, sporty looking",
    "Jail Uniform, black-and-white striped prison uniform, with gang tattoos on the neck",
    "superman costume, with a cape, muscular",
    "gold necklace, muscular, rose tattoo on the chest",
]

for prompt in prompts:
    start_time = time.perf_counter()
    result_image = infer_img2img(prompt=prompt, image=resized_image)
    end_time = time.perf_counter()
    first_word = prompt.split()[0]
    print(f"Inference time for prompt '{prompt}': {end_time - start_time:.2f} seconds")
    result_image.save(f"generated_img_{first_word}.png")
