
import inspect
import re
from typing import Callable, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from packaging import version
from tqdm import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers import SchedulerMixin, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.utils import logging
from PIL import Image
from library import sdxl_model_util, sdxl_train_util, train_util

from library.sdxl_lpw_stable_diffusion import *


from library.hunyuan_models import *


def load_scheduler_sigmas(beta_start=0.00085, beta_end=0.018, num_train_timesteps=1000):
    betas = torch.linspace(beta_start**0.5, beta_end**0.5,
                           num_train_timesteps, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas)
    return alphas_cumprod, sigmas


ATTN_MODE = "xformers"
CLIP_TOKENS = 75 * 2 + 2
DEVICE = "cuda"
try:
    from k_diffusion.external import DiscreteVDDPMDenoiser
    from k_diffusion.sampling import sample_euler_ancestral, get_sigmas_exponential, sample_dpmpp_2m_sde
except ImportError:
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "k-diffusion"])

    from k_diffusion.external import DiscreteVDDPMDenoiser
    from k_diffusion.sampling import sample_euler_ancestral, get_sigmas_exponential, sample_dpmpp_2m_sde


from library.hunyuan_utils import get_cond, calc_rope


class HuanYuanDiffusionLongPromptWeightingPipeline:

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        # clip_skip: int,
        safety_checker,
        feature_extractor,
        requires_safety_checker=False,
        clip_skip=0,
    ):
        # clip skip is ignored currently
        # print("tokenizer: ", tokenizer)
        # print("text_encoder: ", text_encoder)
        self.unet = unet
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor
        self.requires_safety_checker = requires_safety_checker
        self.vae = vae
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.progress_bar = lambda x: tqdm(x, leave=False)

        self.clip_skip = clip_skip
        self.tokenizers = tokenizer
        self.text_encoders = text_encoder

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
            # self.vae.to(device=self.device)
        if dtype is not None:
            self.dtype = dtype

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def check_inputs(self, prompt, height, width, strength, callback_steps):
        pass

    def get_timesteps(self, num_inference_steps, strength, device, is_text2img):
        if is_text2img:
            return self.scheduler.timesteps.to(device), num_inference_steps
        else:
            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(device)
            return timesteps, num_inference_steps - t_start

    def run_safety_checker(self, image, device, dtype):
        return image, None

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents

            image = self.vae.decode(latents.to(self.vae.dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image

    def prepare_extra_step_kwargs(self, generator, eta):
        return {}

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        controlnet=None,
        controlnet_image=None,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):

        BETA_END = 0.018
        CFG_SCALE = guidance_scale
        STEPS = num_inference_steps
        patch_size = 2
        num_heads = 88

        alphas, sigmas = load_scheduler_sigmas(beta_end=BETA_END)
        denoiser = self.unet
        clip_tokenizer = self.tokenizers[0]
        clip_encoder = self.text_encoders[0]

        mt5_embedder = self.text_encoders[1]

        vae = self.vae
        denoiser.eval()
        denoiser.disable_fp32_silu()
        denoiser.disable_fp32_layer_norm()
        denoiser.set_attn_mode(ATTN_MODE)
        vae.requires_grad_(False)
        mt5_embedder.to(torch.float16)

        with torch.autocast("cuda"):
            clip_h, clip_m, mt5_h, mt5_m = get_cond(
                prompt,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                # Should be same as original implementation with max_length_clip=77
                # Support 75*n + 2
                max_length_clip=CLIP_TOKENS,
            )
            neg_clip_h, neg_clip_m, neg_mt5_h, neg_mt5_m = get_cond(
                negative_prompt,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                max_length_clip=CLIP_TOKENS,
            )
            clip_h = torch.concat([clip_h, neg_clip_h], dim=0)
            clip_m = torch.concat([clip_m, neg_clip_m], dim=0)
            mt5_h = torch.concat([mt5_h, neg_mt5_h], dim=0)
            mt5_m = torch.concat([mt5_m, neg_mt5_m], dim=0)
            torch.cuda.empty_cache()

        style = torch.as_tensor([0] * 2, device=DEVICE)
        # src hw, dst hw, 0, 0
        size_cond = [height, width, height, width, 0, 0]
        image_meta_size = torch.as_tensor([size_cond] * 2, device=DEVICE)

        freqs_cis_img = calc_rope(height, width, patch_size, num_heads)

        denoiser_wrapper = DiscreteVDDPMDenoiser(
            # A quick patch for learn_sigma
            lambda *args, **kwargs: denoiser(* \
                                             args, **kwargs).chunk(2, dim=1)[0],
            alphas,
            False,
        ).to(DEVICE)

        def cfg_denoise_func(x, sigma):
            cond, uncond = denoiser_wrapper(
                x.repeat(2, 1, 1, 1),
                sigma.repeat(2),
                encoder_hidden_states=clip_h,
                text_embedding_mask=clip_m,
                encoder_hidden_states_t5=mt5_h,
                text_embedding_mask_t5=mt5_m,
                image_meta_size=image_meta_size,
                style=style,
                cos_cis_img=freqs_cis_img[0],
                sin_cis_img=freqs_cis_img[1],
            ).chunk(2, dim=0)
            return uncond + (cond - uncond) * CFG_SCALE

        sigmas = denoiser_wrapper.get_sigmas(STEPS).to(DEVICE)
        sigmas = get_sigmas_exponential(
            STEPS, denoiser_wrapper.sigma_min, denoiser_wrapper.sigma_max, DEVICE
        )
        x1 = torch.randn(1, 4, height // 8, width // 8,
                         dtype=torch.float16, device=DEVICE)

        with torch.autocast("cuda"):
            sample = sample_dpmpp_2m_sde(
                cfg_denoise_func,
                x1 * sigmas[0],
                sigmas,
            )
            torch.cuda.empty_cache()
        latents = sample
        return latents

    def text2img(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):

        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            callback_steps=callback_steps,
        )

    def latents_to_image(self, latents):
        # 9. Post-processing
        image = self.decode_latents(latents.to(self.vae.dtype))
        image = self.numpy_to_pil(image)
        return image

    # copy from pil_utils.py
    def numpy_to_pil(self, images: np.ndarray) -> Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(
                image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
