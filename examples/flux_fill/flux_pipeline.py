import math
from examples.flux_fill.clip import download_clip_encoder_weights, load_clip_encoder, load_clip_tokenizer
from sampling import (
    denoise,
    get_schedule,
    prepare_fill_model,
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

from PIL import Image
import numpy as np

def numpy_to_pil(images: np.ndarray):
    """
    Convert a numpy image or batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    # If image is grayscale, squeeze the channel dimension and set mode "L"
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def _prepare_noise(height: int, width: int, backend: ml.Backend, dtype: ml.types.Dtype, seed: int):
    """
    Prepare a noise tensor with reshaping and transposing to match expected latent shape.
    """
    # Set noise dimensions based on image height and width (with fixed 16 channels)
    batch_size, channels = 1, 16
    adjusted_height = 2 * math.ceil(height / 16)
    adjusted_width = 2 * math.ceil(width / 16)
    # Create noise using backend's random generator with the given seed
    latents = backend.randn(batch_size, channels, adjusted_height, adjusted_width, key=seed, dtype=dtype)
    # Reshape noise into smaller patches
    latents = backend.reshape(latents, (batch_size, channels, adjusted_height // 2, 2, adjusted_width // 2, 2))
    # Permute dimensions to bring patch dimensions together
    latents = backend.transpose(latents, (0, 2, 4, 1, 3, 5))
    # Flatten to combine patch dimensions and channels
    latents = backend.reshape(latents, (batch_size, -1, channels * 4))
    return latents

def _prepare_latent_image_ids(height: int, width: int, backend: ml.Backend, dtype: ml.types.Dtype, vae_scale_factor: int):
    """
    Prepare latent image IDs for spatial signal conditioning.
    """
    # Calculate scaled image dimensions based on VAE scale factor to ensure proper tiling.
    _height = 2 * (int(height // 2) // (vae_scale_factor * 2))
    _width = 2 * (int(width // 2) // (vae_scale_factor * 2))

    # Create coordinate grids for image IDs:
    #   - First channel: all zeros
    #   - Second channel: row indices repeated across columns
    #   - Third channel: column indices repeated across rows
    img_id_1 = backend.zeros((_height, _width))  
    img_id_2 = backend.arange(_height)[:, None] + backend.zeros((_height, _width))
    img_id_3 = backend.arange(_width)[None, :] + backend.zeros((_height, _width))
    
    # Expand dimensions to allow concatenation
    img_id_1 = img_id_1[..., None]
    img_id_2 = img_id_2[..., None]
    img_id_3 = img_id_3[..., None]
    
    # Concatenate the three channels along the last dimension to form latent IDs
    latent_image_ids = backend.concat([img_id_1, img_id_2, img_id_3], axis=-1)
    # Reshape the grid into a list of position vectors
    latent_image_ids = latent_image_ids.reshape(_height * _width, 3)
    return backend.array(latent_image_ids, dtype=dtype)

def create_pipeline(
    backend: ml.Backend,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    num_steps: int = 5,
    guidance: float = 30.0,
    max_seq_len: int = 512,
):
    # Initialize sub-model names and parameters.
    name = "flux-dev-fill"
    
    print("Loading T5 Encoder")
    t5_lm = load_t5_encoder(name, max_seq_len)
    t5_tokenizer = load_t5_tokenizer(backend, max_seq_len, name)
    t5_weights = download_t5_encoder_weights(backend, name)
    t5_lm.name = "t5"
    
    print("Loading CLIP Encoder")
    clip_lm = load_clip_encoder(name)
    clip_tokenizer = load_clip_tokenizer(backend, name)
    clip_weights = download_clip_encoder_weights(backend, name)
    clip_lm.name = "clip"
    clip_lm.set_cout("output")
    
    print("Loading AutoEncoder")
    encoder_lm, encoder_params = load_encoder(name, backend, width, height, seed)
    encoder_lm.name = "encoder"

    # Prepare the fill model using the T5, CLIP, and AutoEncoder models.
    prepare_lm = prepare_fill_model(t5=t5_lm, clip=clip_lm, encoder=encoder_lm)
    prepare_pm = ml.compile(
        model=prepare_lm,
        backend=backend,
        jit=True,
        inference=True,
        use_short_namings=False
    )
    
    # Load the flow model which is later used for denoising.
    print("Loading Flow Model")
    flow_pm, _, flow_params = load_flow_model(name, backend=backend, height=height, width=width)
    
    # Load the decoder model to convert processed latent representations back to image space.
    print("Loading Decoder Model\n")
    decoder_pm, _, decoder_params = load_decoder(name, backend, width, height)
    
    vae_scale_factor = 8  # fixed VAE scale factor; ideally should come from config
    # Determine the sequence length of the image based on the down-scale factor
    image_seq_len = (int(height) // vae_scale_factor // 2) * (int(width) // vae_scale_factor // 2)
    # Obtain a denoising schedule (timesteps) based on the number of steps and image sequence length
    timesteps = get_schedule(num_steps, image_seq_len, shift=True, backend=backend)

    # Prepare a combined dictionary of all model parameters.
    encoder_params = {f"encoder_{key}": value for key, value in encoder_params.items()}
    t5_params = {f"t5_{key}": value for key, value in t5_weights.items()}
    clip_params = {f"clip_{key}": value for key, value in clip_weights.items()}
    all_params = {**encoder_params, **t5_params, **clip_params}

    def pipeline_fn(prompt, img_cond, img_mask):
        """
        Given a text prompt and images for conditioning and mask, generate a filled image.
        """
        # Convert the conditions and mask images to numpy arrays.
        img_cond = np.array(img_cond.convert("RGB"))
        img_mask = np.array(img_mask.convert("L"))

        # Tokenize prompt for both CLIP and T5 encoders.
        clip_inp = clip_tokenizer.encode(prompt)
        t5_inp = t5_tokenizer.encode(prompt)
        model_inputs = {
            "t5_input": t5_inp,
            "clip_input": clip_inp,
            "image": backend.array(img_cond, dtype=ml.bfloat16),
            "mask_image": backend.array(img_mask, dtype=ml.bfloat16)
        }
        # Run the core pipeline that prepares, denoises, unpacks and finally decodes.
        # Run the fill model to prepare features.
        prepare_outputs = prepare_pm.evaluate(all_params, model_inputs, state=prepare_pm.initial_state_dict)
        
        # Generate latent image ids and noise then update the outputs.
        latent_image_ids = _prepare_latent_image_ids(height, width, backend, ml.bfloat16, vae_scale_factor)
        prepare_outputs["latent_image_ids"] = latent_image_ids.reshape(1, *latent_image_ids.shape)
        prepare_outputs["latents"] = _prepare_noise(height, width, backend, ml.bfloat16, seed)

        # Denoise and unpack the latent representation.
        data = denoise(flow_pm, flow_params, timesteps=timesteps, backend=backend, guidance=guidance, **prepare_outputs)
        unpacked_input = unpack(data, height, width, backend)
        output = decoder_pm.evaluate(decoder_params, {"input": unpacked_input})

        # Denormalize the output image tensor from [-1, 1] to [0, 1] and cast for display.
        _output = backend.clip(output["output"] * 0.5 + 0.5, 0, 1)
        _output = backend.cast(_output, ml.float)
        _output = backend.to_device(_output, "cpu")
        np_out = np.transpose(np.array(_output), (0, 2, 3, 1))
        # Convert the numpy image to a PIL image and return the first (or only) image.
        return numpy_to_pil(np_out)[0]
    
    return pipeline_fn