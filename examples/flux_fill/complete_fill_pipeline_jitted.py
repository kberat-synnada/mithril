# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from clip import download_clip_encoder_weights, load_clip_encoder, load_clip_tokenizer
from sampling import (
    denoise,
    get_noise,
    get_schedule,
    prepare_fill,
    unpack,
)
from diffusers import FluxFillPipeline
FluxFillPipeline.from_pretrained
import jax
import jax.profiler
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


from t5 import download_t5_encoder_weights, load_t5_encoder, load_t5_tokenizer
from util import load_decoder, load_encoder, load_flow_model

import mithril as ml
from mithril.models import Model

from PIL import Image
import numpy as np

def numpy_to_pil(images: np.ndarray):
    r"""
    Convert a numpy image or a batch of images to a PIL image.

    Args:
        images (`np.ndarray`):
            The image array to convert to PIL format.

    Returns:
        `List[PIL.Image.Image]`:
            A list of PIL images.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def _prepare_latent_image_ids(height, width, backend, dtype):
    # Create each component
    img_id_1 = backend.zeros((height, width))  # First channel
    img_id_2 = backend.arange(height)[:, None] + backend.zeros((height, width))  # Second channel
    img_id_3 = backend.arange(width)[None, :] + backend.zeros((height, width))  # Third channel
    
    # Add new axis to each for concatenation
    img_id_1 = img_id_1[..., None]
    img_id_2 = img_id_2[..., None]
    img_id_3 = img_id_3[..., None]
    
    # Concatenate along the last dimension
    latent_image_ids = backend.concat([img_id_1, img_id_2, img_id_3], axis=-1)
    
    # Reshape
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    
    return backend.array(latent_image_ids, dtype=dtype)

def get_pipeline_fn(
    # img_cond,
    # img_mask,
    backend,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    # prompt: str = "muscular, wider shoulder,bigger arms,sporty looking",
    num_steps: int = 5,
    guidance: float = 30.0,
    max_seq_len: int = 512,
):
    # Create required sub-models
    name = "flux-dev-fill"
    print("Loading T5 encoder")
    t5_lm = load_t5_encoder(name, max_seq_len)
    t5_tokenizer = load_t5_tokenizer(backend, max_seq_len, name)  # noqa F841
    t5_weights = download_t5_encoder_weights(backend, name)  # noqa F841
    t5_lm.name = "t5"
    
    print("Loading CLIP encoder")
    clip_lm = load_clip_encoder(name)
    clip_tokenizer = load_clip_tokenizer(backend, name)  # noqa F841
    clip_weights = download_clip_encoder_weights(backend, name)  # noqa F841
    clip_lm.name = "clip"
    clip_lm.set_cout("output")
    
    print("Loading AutoEncoder")
    encoder_lm, encoder_params = load_encoder(name,backend, width, height)  # noqa F841
    encoder_lm.name = "encoder"

    # Prepare parameters
    encoder_params = {f"encoder_{key}": value for key, value in encoder_params.items()}
    t5_params = {f"t5_{key}": value for key, value in t5_weights.items()}
    clip_params = {f"clip_{key}": value for key, value in clip_weights.items()}
    all_params = {**encoder_params, **t5_params, **clip_params}
    
    print("Generate Noise")
    noise = get_noise(
        1,
        height,
        width,
        seed,
    )

    kwargs = prepare_fill(t5=t5_lm, clip=clip_lm, noise=noise, encoder=encoder_lm)
    prepare_lm = Model.create(**{f"_{key}": value for key, value in kwargs.items()})
    prepare_pm = ml.compile(
        model = prepare_lm,
        backend = backend,
        jit = False,
        inference = True,
        use_short_namings=False
    )
    # Load flow model
    print("Loading Flow model")
    flow_pm, _, flow_params = load_flow_model(name, backend=backend, height=height, width=width)  # noqa F841

    # Load flow model
    print("Loading Decoder model")
    decoder_pm, _, decoder_params = load_decoder(name, backend, width, height)  # noqa F841

    print("get_schedule")
    vae_scale_factor = 8 # TODO: get from config
    image_seq_len = (int(height) // vae_scale_factor // 2) * (int(width) // vae_scale_factor // 2)
    timesteps = get_schedule(
        num_steps,
        image_seq_len,
        shift=True,
        backend=backend,
    )
    def _pipeline_core(model_inputs):
        prepare_outputs = prepare_pm.evaluate(all_params, model_inputs, state=prepare_pm.initial_state_dict)
        # prepare_outputs = prepare_pm.evaluate(all_params, model_inputs, state=prepare_pm.initial_state_dict)
        prepare_outputs = {key[1:]: value for key, value in prepare_outputs.items()}
        _height = 2 * (int(height) // (vae_scale_factor * 2))
        _width = 2 * (int(width) // (vae_scale_factor * 2))
        latent_image_ids = _prepare_latent_image_ids(_height // 2, _width // 2, backend, ml.bfloat16)
        prepare_outputs["latent_image_ids"] = latent_image_ids.reshape(1, *latent_image_ids.shape)

        data = denoise(flow_pm, flow_params, timesteps=timesteps, backend=backend, guidance=guidance, **prepare_outputs)
        # data = denoise(flow_pm, flow_params, timesteps=timesteps, backend=backend, **inputs)
        unpacked_input = unpack(data, height, width, backend)
        output = decoder_pm.evaluate(decoder_params, {"input": unpacked_input})
        return output
    # jitted_pipeline_core = backend.jit(_pipeline_core)

    def pipeline_fn(prompt, img_cond, img_mask):
        img_cond = np.array(img_cond.convert("RGB"))
        img_mask = np.array(img_mask.convert("L"))

        clip_inp = clip_tokenizer.encode(prompt)
        t5_inp = t5_tokenizer.encode(prompt)
        model_inputs = {
            "prompt_embeds": t5_inp,
            "pooled_prompt_embeds": clip_inp,
            "image": backend.array(img_cond, dtype = ml.bfloat16),
            "mask_image":  backend.array(img_mask, dtype = ml.bfloat16)
        }        
        output = _pipeline_core(model_inputs)

        print("Denormalize output")
        _output = backend.clip(output["output"] * 0.5 + 0.5, 0, 1)
        _output = backend.cast(_output, ml.float)
        _output = backend.to_device(_output, "cpu")
        np_out = np.transpose(np.array(_output), (0, 2, 3, 1))
        return numpy_to_pil(np_out)[0]
    return pipeline_fn