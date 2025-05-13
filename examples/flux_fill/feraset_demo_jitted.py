from dataclasses import dataclass
from typing import Optional

import random
import requests
from io import BytesIO
import numpy as np
import PIL
from PIL import Image

import mithril as ml
from complete_fill_pipeline_jitted import get_pipeline_fn

import jax
import jax.profiler
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

MIN_ASPECT_RATIO = 9 / 16
MAX_ASPECT_RATIO = 16 / 9
FIXED_DIMENSION = 1024

@dataclass
class Config:
    device: str
    model_id: str
    num_inference_steps: int
    width: int
    height: int
    dtype: type
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



import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
import mithril as ml

from segformer_semantic_segmentation import segformer_semantic_segmentation
import time


def load_weights(
    param_shapes, torch_model, backend: ml.Backend
):
    ml_params = {}
    torch_state_dict = torch_model.state_dict()

    for torch_key in torch_state_dict:
        ml_key = torch_key.replace(".", "_").lower()
        if ml_key not in param_shapes:
            continue

        param_shape = param_shapes[ml_key]
        parameter = torch_state_dict[torch_key].numpy().reshape(param_shape)
        ml_params[ml_key] = backend.array(parameter)

    return ml_params

SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
# PROMPT = "muscular, wider shoulder,bigger arms,sporty looking"
PROMPT = "prisoner clothing, with gang tattoos"
ITERATION = 20

config = Config(
    model_id="black-forest-labs/FLUX.1-Fill-dev",
    dtype=ml.bfloat16,
    device="tpu",
    guidance_scale=50,
    num_inference_steps=30,
    max_sequence_length=512,
    width=1024,
    height=1536,
)

processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)

# Mithril segformer model and backend.
mitihril_model = segformer_semantic_segmentation(model.segformer.config)
# backend = ml.TorchBackend(device="cuda")
backend = ml.JaxBackend(device="tpu", dtype=ml.bfloat16, device_mesh=(4,))
# Compile mithril model.
input_shape = [1, 3, 512, 512]
pm = ml.compile(
    mitihril_model, 
    backend=backend,
    shapes={"input": input_shape},
    data_keys={"input"},
    use_short_namings=False,
    safe_names=False,
)
# Load weights from torch model to mithril model.
params = load_weights(pm.shapes, model, backend)
# Prepare data for Mithril model.
data = {
    "decode_head_batch_norm_running_mean": params.pop(
        "decode_head_batch_norm_running_mean"
    ),
    "decode_head_batch_norm_running_var": params.pop(
        "decode_head_batch_norm_running_var"
    ),
}

def segformer_seg(reference_image):
    inputs = processor(images=reference_image, return_tensors='np')
    data["input"] = backend.array(inputs["pixel_values"])
    # Run segmentation model.
    upsampled_logits = pm.evaluate(params, data)["upsampled_logits"]
    # upsampled_logits = backend.to_device(upsampled_logits, "cpu")
    upsampled_logits = backend.to_device(upsampled_logits, "cpu")
    pred_seg = np.array(backend.argmax(upsampled_logits, axis=1)[0])

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
num_steps = 10
pipeline = get_pipeline_fn(
    backend=backend,
    guidance=config.guidance_scale,
    width=1024,
    height=1024,
    num_steps=num_steps,
    max_seq_len=512,
)

def infer_img2img(prompt, image) -> Image:
    resized_image = resize_img(image, max_side=1024)
    mask = segformer_seg(resized_image)
    final_mask = mask.resize(resized_image.size)
    # width, height = calculate_optimal_dimensions(resized_image)
    muscle_image = pipeline(
        prompt=prompt,
        img_cond=resized_image,
        img_mask=final_mask,
    )
    return muscle_image


seed = random.randint(0, 2**63 - 1)

input_img_url = "https://i.ibb.co/xKS4F80b/leonardo-dicaprio-1.jpg"
response = requests.get(input_img_url)
image = Image.open(BytesIO(response.content)).convert("RGB")
# path = "examples/flux_fill/emre.jpeg"
# image = Image.open(path).convert("RGB")
# image = Image.open(sys.argv[1]).convert("RGB")

start_time = time.perf_counter()
result_image = infer_img2img(
    prompt="muscular, wider shoulder,bigger arms,sporty looking",
    image=image,
)
end_time = time.perf_counter()
print(f"Inference time: {end_time - start_time:.2f} seconds")
result_image.save("generated_img_muscle.png")

start_time = time.perf_counter()
result_image = infer_img2img(
    prompt='prisoner clothing, with gang tattoos',
    image=image,
)
end_time = time.perf_counter()
print(f"Second Inference time: {end_time - start_time:.2f} seconds")
result_image.save("generated_img_prisoner.png")
