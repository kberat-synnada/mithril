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


import os
import math
from dataclasses import dataclass

import torch
from einops import rearrange
from safetensors import safe_open
from conditioner import HFEmbedder
from imwatermark import WatermarkEncoder
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSemanticSegmentation
from auto_encoder import AutoEncoderParams, decode, encode

from model import FluxParams, flux
from segformer_semantic_segmentation import segformer_semantic_segmentation

import mithril as ml


def load_t5(device: str | torch.device = "cuda", max_length: int = 128) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder(
        "google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16
    ).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder(
        "openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16
    ).to(device)


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


def load_seg_model(
    model_id, backend: ml.Backend
):
    t_model = AutoModelForSemanticSegmentation.from_pretrained(model_id)

    # Mithril segformer model and backend.
    segformer_model = segformer_semantic_segmentation(t_model.segformer.config)

    # Compile segformer model.
    segformer_pm = ml.compile(
        segformer_model, 
        backend=backend,
        shapes={"input": [1, 3, 512, 512]},
        data_keys={"input"},
        use_short_namings=False,
        safe_names=False,
    )

    ml_params = {}
    torch_state_dict = t_model.state_dict()
    param_shapes = segformer_pm.shapes
    for torch_key in torch_state_dict:
        ml_key = torch_key.replace(".", "_").lower()
        if ml_key not in param_shapes:
            continue

        param_shape = param_shapes[ml_key]
        parameter = torch_state_dict[torch_key].numpy().reshape(param_shape)
        ml_params[ml_key] = backend.array(parameter)

    return segformer_pm, ml_params

def convert_to_ml_weights(
    model: ml.models.PhysicalModel,
    sd,
    backend: ml.Backend,
    dtype: ml.types.Dtype | None = None,
):
    params = {}
    ml_param_shapes = model.shapes
    shards = model.propose_shardings()
    for k in sd.keys():  # type: ignore #noqa SIM118
        ml_key = k.replace(".", "_").lower()
        if ml_key not in ml_param_shapes:
            # print(f"Skipping {k}")
            continue
        if isinstance(sd, safe_open):
            param = sd.get_tensor(k)  # type: ignore
        else:
            param = sd[k]
        param_shape = ml_param_shapes[ml_key]
        params[ml_key] = backend.reshape(param, param_shape)
        mesh = None
        if math.prod(param_shape) > 512 * 512 * 16:
            mesh = shards[ml_key]
        params[ml_key] = backend.array(params[ml_key], dtype=dtype, device_mesh=mesh)
    return params

configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
            max_seq_len=128
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
            max_seq_len=128
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fill": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FILL"),
        params=FluxParams(
            in_channels=384,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
            max_seq_len=512
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def load_flow_model(name: str, backend: ml.Backend, height, width, hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and (r_id := configs[name].repo_id) is not None
        and (r_flow := configs[name].repo_flow) is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(r_id, r_flow)

    flux_lm = flux(configs[name].params, height, width)
    flux_pm = ml.compile(
        flux_lm,
        backend,
        jit=True,
        use_short_namings=False,
    )

    assert ckpt_path is not None
    sd = safe_open(ckpt_path, "pt" if backend.backend_type == "torch" else "jax", "cpu")
    params = convert_to_ml_weights(flux_pm, sd, backend)

    return flux_pm, flux_lm, params


def load_decoder(
    name: str, backend: ml.Backend, width:int, height:int, hf_download: bool = True
) -> tuple[ml.models.PhysicalModel, ml.models.Model, dict]:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and (r_id := configs[name].repo_id) is not None
        and (r_ae := configs[name].repo_ae) is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(r_id, r_ae)

    # Loading the autoencoder
    decoder_lm = decode(configs[name].ae_params)
    decoder_lm.set_shapes(input=[1, 16, height // 8, width // 8])

    decoder_pm = ml.compile(
        decoder_lm,
        backend=backend,
        inference=True,
        jit=True,
        data_keys=["input"],
        use_short_namings=False,
    )

    assert ckpt_path is not None

    sd = safe_open(ckpt_path, "pt" if backend.backend_type == "torch" else "jax", "cpu")
    params = convert_to_ml_weights(decoder_pm, sd, backend, ml.bfloat16)

    return decoder_pm, decoder_lm, params


def load_encoder(
    name: str, backend: ml.Backend, width:int, height:int, hf_download: bool = True, seed: int = 42
) -> tuple[ml.models.Model, dict]:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and (r_id := configs[name].repo_id) is not None
        and (r_ae := configs[name].repo_ae) is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(r_id, r_ae)

    # Loading the autoencoder
    encoder_lm = encode(seed, configs[name].ae_params)
    encoder_lm.set_shapes(input=[1, 3, height, width])

    encoder_pm = ml.compile(
        encoder_lm,
        backend=backend,
        inference=True,
        jit=True,
        data_keys=["input"],
        use_short_namings=False,
    )

    assert ckpt_path is not None

    sd = safe_open(ckpt_path, "pt" if backend.backend_type == "torch" else "jax", "cpu")
    params = convert_to_ml_weights(encoder_pm, sd, backend, ml.bfloat16)

    return encoder_lm, params


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange(
            (255 * image).detach().cpu(), "n b c h w -> (n b) h w c"
        ).numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(
            rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)
        ).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


# A fixed 48-bit message that was chosen at random
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
