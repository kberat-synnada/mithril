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

import random
import math
from collections.abc import Callable
from copy import deepcopy

import mithril as ml
from mithril.models import (
    Arange,
    BroadcastTo,
    Concat,
    IOKey,
    Multiply,
    Ones,
    Randn,
    Reshape,
)


def prepare_logical(
    block: ml.models.Model,
    t5: ml.models.Model,
    clip: ml.models.Model,
    num_samples: int,
    height: int,
    width: int,
):
    c = 16
    h = 2 * math.ceil(height / 16)
    w = 2 * math.ceil(width / 16)

    block |= Randn(shape=(num_samples, (h // 2) * (w // 2), c * 2 * 2)).connect(
        output=IOKey("img")
    )

    block |= Ones(shape=(num_samples, h // 2, w // 2, 1)).connect(output="ones")
    block |= Multiply().connect(left="ones", right=0, output="img_ids_preb")
    block |= Arange(stop=(w // 2)).connect(output="arange_1")
    block |= BroadcastTo(shape=(num_samples, h // 2, w // 2)).connect(
        block.arange_1[None, :, None],  # type: ignore
        output="arange_1_bcast",
    )
    block |= Arange(stop=(h // 2)).connect(output="arange_2")
    block |= BroadcastTo(shape=(num_samples, h // 2, w // 2)).connect(
        block.arange_2[None, None, :],  # type: ignore
        output="arange_2_bcast",
    )
    block |= Concat(axis=-1).connect(
        input=[
            block.img_ids_preb,  # type: ignore
            block.arange_1_bcast[..., None],  # type: ignore
            block.arange_2_bcast[..., None],  # type: ignore
        ],
        output="img_ids_cat",
    )

    block |= Reshape(shape=(num_samples, -1, 3)).connect(
        block.img_ids_cat,  # type: ignore
        output=IOKey("img_ids"),
    )

    block |= t5.connect(input=IOKey("t5_tokens"), output=IOKey("txt"))
    block |= Ones().connect(
        shape=(num_samples, block.txt.shape[1], 3),  # type: ignore
        output="txt_ids_preb",
    )
    block |= Multiply().connect(left="txt_ids_preb", right=0, output=IOKey("txt_ids"))

    block |= clip.connect(input=IOKey("clip_tokens"), output=IOKey("y"))


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
    *,
    backend: ml.Backend,
) -> list[float]:
    # extra step for zero
    timesteps = backend.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def unpack(
    input: ml.models.Connection, height: int, width: int, backend: ml.Backend
) -> ml.models.Connection:
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    b = input.shape[0]

    input = input.reshape((b, h, w, -1, 2, 2))
    input = backend.transpose(input, (0, 3, 1, 4, 2, 5))
    input = input.reshape((b, -1, 2 * h, 2 * w))

    return input

def denoise(
    model: ml.models.PhysicalModel,
    params: dict,
    # model input
    latents,
    masked_image_latents,    
    latent_image_ids,
    prompt_embeds,
    text_ids,
    pooled_prompt_embeds,
    # sampling parameters
    timesteps: list[float],
    backend: ml.Backend,
    guidance: float,
):
    # this is ignored for schnell
    latents_shp = latents.shape[0]
    guidance_vec = backend.ones((latents_shp,)) * guidance
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=False):
        t_vec = backend.ones((latents_shp,)) * t_curr
        pred = model.evaluate(
            params,
            {
                "img": backend.concat((latents, masked_image_latents), axis=2),
                "img_ids": latent_image_ids,
                "txt": prompt_embeds,
                "txt_ids": text_ids,
                "y": pooled_prompt_embeds,
                "timesteps": t_vec,
                "guidance": guidance_vec,
            },
        )

        latents = latents + (t_prev - t_curr) * pred["output"]  # type: ignore[operator]

    return latents

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    seed: int,
) -> ml.models.Connection:
    height = 2 * math.ceil(height / 16)
    width = 2 * math.ceil(width / 16)
    noise_model = Randn(shape=(num_samples, 16, height, width), key= random.randint(0, 2**63 - 1))
    # noise_model = Ones(shape=(num_samples, 16, height, width))
    return noise_model.output


def prepare_fill(
    t5: ml.models.Model,
    clip: ml.models.Model,
    noise: ml.models.Connection,
    encoder: ml.models.Model,
):

    prompt_embeds = t5(input = IOKey("prompt_embeds"))
    text_ids = Ones()(shape=(1, prompt_embeds.shape[1], 3)) * 0.0
    pooled_prompt_embeds = clip(input = IOKey("pooled_prompt_embeds"))
    
    encode_mask_model = deepcopy(encoder)
    encode_mask_model.name = "encoder"
    encoder.name = "encoder_cond"
    
    encoder_kwargs = {
        key: IOKey()
        for key in encode_mask_model.input_keys
        if "$" in key
    }

    # TODO: implement encoder for mask
    # image = IOKey("image", shape=[1024, 1024, 3])
    image = IOKey("image", shape=[None, None, 3])
    init_image = image / 127.5 - 1.0
    init_image = init_image.transpose((2, 0, 1))[None, ...]


    bs = noise.shape[0]
    h = noise.shape[2]
    w = noise.shape[3]

    img_id_1 = Ones()(shape=(h // 2, w // 2)) * 0.0
    
    img_id_2 = Arange()(stop=(h // 2))[:, None]
    img_id_2 = BroadcastTo()(input = img_id_2, shape = (h // 2, w // 2))
    
    img_id_3 = Arange()(stop=(w // 2))[None, :]
    img_id_3 = BroadcastTo()(input = img_id_2, shape = (h // 2, w // 2))
    
    latent_image_ids = Concat(axis=-1)(input = [img_id_1[..., None], img_id_2[..., None], img_id_3[..., None]])
    
    latent_image_ids = latent_image_ids.reshape((bs, -1, 3))
    image_latents = encoder(input=init_image, **encoder_kwargs)
    latents = noise
    
    b_cond = latents.shape[0]
    c_cond = latents.shape[1]
    h_cond = latents.shape[2]
    w_cond = latents.shape[3]

    # rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    latents = latents.reshape((b_cond, c_cond, h_cond // 2, 2, w_cond // 2, 2))
    latents = latents.transpose((0, 2, 4, 1, 3, 5))
    latents = latents.reshape((b_cond, -1, c_cond * 4))
    
    
    
    ###### MASK #####
    # mask_image = IOKey("mask_image", shape = [1024, 1024])
    mask_image = IOKey("mask_image", shape = [None, None])
    mask_image = mask_image / 255.0
    masked_image = init_image * (1 - mask_image)
    masked_image_latents = encode_mask_model(input = masked_image, **encoder_kwargs)

    b_masked = masked_image_latents.shape[0]
    c_masked = masked_image_latents.shape[1]
    h_masked = masked_image_latents.shape[2]
    w_masked = masked_image_latents.shape[3]
    
    masked_image_latents = masked_image_latents.reshape((b_masked, c_masked, h_masked // 2, 2, w_masked // 2, 2))
    masked_image_latents = masked_image_latents.transpose((0, 2, 4, 1, 3, 5))
    masked_image_latents = masked_image_latents.reshape((b_masked,- 1, c_masked * 4))
    
    mask_image = mask_image[None, None, ...]
    mask_image = mask_image[:, 0, :, :]
    
    mask_image = mask_image.reshape((b_masked, h_masked, 8, w_masked, 8))
    mask_image = mask_image.transpose((0, 2, 4, 1, 3))
    mask_image = mask_image.reshape((b_masked, 64, h_masked, w_masked))
    
    mask_image = mask_image.reshape((b_masked, 64, h_masked // 2, 2,  w_masked // 2, 2))
    mask_image = mask_image.transpose((0, 2, 4, 1, 3, 5))
    mask = mask_image.reshape((b_masked, -1, 64 * 4))
    masked_image_latents = Concat(axis=-1)(input = [masked_image_latents, mask])

    kwargs = {}
    kwargs["text_ids"] = text_ids
    kwargs["latent_image_ids"] = latent_image_ids
    kwargs["latents"] = latents
    kwargs["masked_image_latents"] = masked_image_latents
    kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
    kwargs["prompt_embeds"] = prompt_embeds
    
    return kwargs


def denoise_logical(
    flux_model: ml.models.Model,
    masked_image_latents: ml.models.Connection,
    latent_image_ids: ml.models.Connection,
    prompt_embeds: ml.models.Connection,
    text_ids: ml.models.Connection,
    pooled_prompt_embeds: ml.models.Connection,
    timesteps: list[float],
    latents: ml.models.Connection,
    guidance: float = 4.0,

    
):
    guidance_vec = Ones()(shape=[latents.shape[0]]) * guidance

    weigth_kwargs = {
        key: IOKey()
        for key in flux_model.input_keys
        if "$" in key
    }
    i = 0
    for t_prev, t_curr in zip(timesteps[:-1], timesteps[1:]):
        print(f"denoise step {i}")
        t_vec = Ones()(shape=[latents.shape[0]]) * t_prev
        _flux_model = deepcopy(flux_model)

        print(f"latents shape: {latents.metadata.shape.get_shapes()}")
        print(f"masked_image_latents shape: {masked_image_latents.metadata.shape.get_shapes()}")
        print(f"latent_image_ids shape: {latent_image_ids.metadata.shape.get_shapes()}")
        print(f"text_ids shape: {text_ids.metadata.shape.get_shapes()}")
        print(f"pooled_prompt_embeds shape: {pooled_prompt_embeds.metadata.shape.get_shapes()}")
        print(f"t_vec shape: {t_vec.metadata.shape.get_shapes()}")
        print(f"guidance shape: {guidance_vec.metadata.shape.get_shapes()}")

        pred = _flux_model(
            img=Concat(axis = -1)((latents, masked_image_latents)),
            img_ids=latent_image_ids,
            txt=prompt_embeds,
            txt_ids=text_ids,
            y=pooled_prompt_embeds,
            timesteps=t_vec,
            guidance=guidance_vec,
            **weigth_kwargs,
        )
        latents = latents + (t_curr - t_prev) * pred
        i += 1

    return pred
