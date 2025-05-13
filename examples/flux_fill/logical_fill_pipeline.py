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
    denoise_logical,
    get_noise,
    get_schedule,
    prepare_fill,
    unpack,
)
from diffusers import FluxFillPipeline
FluxFillPipeline.from_pretrained


from t5 import download_t5_encoder_weights, load_t5_encoder, load_t5_tokenizer
from util import load_decoder, load_encoder, load_flow_model

import mithril as ml
from mithril.models import (
    IOKey,
    Model
)
import pickle

from PIL import Image
import numpy as np

from feraset_test import resize_img, segformer_seg, calculate_optimal_dimensions
import requests
from io import BytesIO
import torch


# def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
#     r"""
#     Convert a PyTorch tensor to a NumPy image.

#     Args:
#         images (`torch.Tensor`):
#             The PyTorch tensor to convert to NumPy format.

#     Returns:
#         `np.ndarray`:
#             A NumPy array representation of the images.
#     """
#     images = (images.cpu().float().numpy())
#     return images

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

# loaded_tensor = torch.load('decoded_tensor.pt')
# output = (loaded_tensor * 0.5 + 0.5).clamp(0, 1).cpu()
# np_out = output.cpu().permute(0, 2, 3, 1).float().numpy()
# img = numpy_to_pil(np_out)
# img[0].save("last_image.png")


def pipeline(
    img_cond,
    img_mask,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    prompt: str = "beautiful woman with nice brests",
    device: str = "cuda",
    num_steps: int = 5,
    guidance: float = 30.0,
    output_dir: str = "output",
    max_seq_len: int = 512
):
        
    backend = ml.TorchBackend(device=device, dtype=ml.bfloat16)
        
    img_cond = np.array(img_cond.convert("RGB"))
    img_mask = np.array(img_mask.convert("L"))

    height = img_cond.shape[0]
    width = img_cond.shape[1]


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

    decoder_pm, decoder_lm, decoder_params = load_decoder(name, backend, width, height)  # noqa F841
    decoder_lm.name = "decoder"
    decoder_pm.evaluate(decoder_params, {"input": torch.load("decode_input.pt").to(torch.bfloat16).to("cuda")})

    encoder_lm, encoder_params = load_encoder(name,backend, width, height)  # noqa F841
    encoder_lm.name = "encoder"

    flow_lm, flow_params = load_flow_model(name, backend=backend)  # noqa F841
    
    clip_inp = clip_tokenizer.encode(prompt)
    t5_inp = t5_tokenizer.encode(prompt)
    
    print(f"tokenizer: {t5_inp.sum()}")
    
    model_inputs = {
        "prompt_embeds": t5_inp,
        "pooled_prompt_embeds": clip_inp,
        "image": backend.array(img_cond, dtype = ml.bfloat16),
        "mask_image":  backend.array(img_mask, dtype = ml.bfloat16)
    }
    
    flow_params = {f"model_0_{key}": value for key, value in flow_params.items()}
    decoder_params = {f"decoder_{key}": value for key, value in decoder_params.items()}
    encoder_params = {f"encoder_{key}": value for key, value in encoder_params.items()}
    t5_params = {f"t5_{key}": value for key, value in t5_weights.items()}
    clip_params = {f"clip_{key}": value for key, value in clip_weights.items()}
    
    all_params = {**flow_params, **decoder_params, **encoder_params, **t5_params, **clip_params}
    
    print("get_noise")
    noise = get_noise(
        1,
        height,
        width,
        seed,
    )

    print("prep fill")
    kwargs = prepare_fill(t5=t5_lm, clip=clip_lm, noise=noise, encoder=encoder_lm)
    # model = Model.create(**{f"_{key}": value for key, value in kwargs.items()})
    # model_pm = ml.compile(
    #     model = model,
    #     backend = backend,
    #     jit = False,
    #     inference = True,
    #     use_short_namings=False
    # )
    # all_params = {**encoder_params, **t5_params, **clip_params}
    # output = model_pm.evaluate(all_params, model_inputs)
    # for key, value in output.items():
    #     print(f"{key} -> {value.sum()}")
    # ...


    print("get_schedule")
    timesteps = get_schedule(
        num_steps,
        kwargs["latents"].metadata.shape.get_shapes()[1],  # type: ignore
        shift=True,
        backend=backend,
    )
    
    denoise_output = denoise_logical(
        flow_lm,
        timesteps=timesteps,
        guidance=guidance,
        **kwargs
    )
    print("unpack_output")
    unpacked_output = unpack(denoise_output, height, width)
    print("decode_output")
    
    print(f"unpacked output: {unpacked_output.metadata.shape.get_shapes()}")
    print(f"decocde model input: {decoder_lm.cin.metadata.shape.get_shapes()}")
    decoded_output = decoder_lm(input = unpacked_output)

    print("create_model")
    finalized_model = ml.models.Model.create(output=decoded_output.transpose((0, 2, 3, 1)))
    
    print("compile")
    denoise_pm = ml.compile(
        finalized_model, backend, inference=True, jit=False, use_short_namings=False
    )
    print("evaluate")
    output = denoise_pm.evaluate(all_params, model_inputs)
    # loaded_tensor = torch.load('decoded_tensor.pt')
    _output = (output["output"] * 0.5 + 0.5).clamp(0, 1).cpu()
    # np_out = _output.cpu().permute(0, 2, 3, 1).float().numpy()
    np_out = _output.cpu().float().numpy()
    img = numpy_to_pil(np_out)
    img[0].save("last_image1.png")

    # torch.save(output["output"], 'tensor.pt')
    
    # _output = output["output"].float().cpu()
    # output = numpy_to_pil(pt_to_numpy(_output))
    # output[0].save("last_image.png")
    
    # img_pil.save("img.png")
    # img_pil = Image.fromarray(
    #     np.array(127.5 * (output["output"].float().cpu()[0] + 1.0)).clip(0, 255).astype(np.uint8)  # type: ignore
    # )
    
    # # img_pil.save("img.png")
    # return img_pil


input_img_url = "https://i.ibb.co/xKS4F80b/leonardo-dicaprio-1.jpg"
response = requests.get(input_img_url)
image = Image.open(BytesIO(response.content)).convert("RGB")
resized_image = resize_img(image, max_side=1024)
mask = segformer_seg(resized_image)
final_mask = mask.resize(resized_image.size)
width, height = calculate_optimal_dimensions(resized_image)
muscle_image = pipeline(
    img_cond=resized_image,
    img_mask=final_mask,
    guidance=50,
    width=1024,
    height=1024,
    num_steps=2,
    max_seq_len=512,
)


