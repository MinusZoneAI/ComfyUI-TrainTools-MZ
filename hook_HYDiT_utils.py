

import json
import os
import time
import torch


UNET_PATH = "ckpts/t2i/model/pytorch_model_ema.pt"


def set_unet_path(path):
    global UNET_PATH
    UNET_PATH = path


VAE_EMA_PATH = "ckpts/t2i/sdxl-vae-fp16-fix"


def set_vae_ema_path(path):
    global VAE_EMA_PATH
    VAE_EMA_PATH = path


TOKENIZER = "ckpts/t2i/tokenizer"


def set_tokenizer_path(path):
    global TOKENIZER
    TOKENIZER = path


TEXT_ENCODER = 'ckpts/t2i/clip_text_encoder'


def set_text_encoder_path(path):
    global TEXT_ENCODER
    TEXT_ENCODER = path


T5_ENCODER = {
    'MT5': None,
    'attention_mask': True,
    'layer_index': -1,
    'attention_pool': True,
    'torch_dtype': torch.float16,
    'learnable_replace': True
}


def set_t5_encoder_path(path):
    global T5_ENCODER
    T5_ENCODER['MT5'] = path


global TRAIN_CONFIG


def set_train_config(train_config):
    global TRAIN_CONFIG
    TRAIN_CONFIG = train_config


def easy_sample_images(
        args,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        model=None,
        embedder_t5=None,
        target_height=768,
        target_width=1280,
        prompt="A photo of a girl with a hat on a sunny day",
        negative_prompt="",
        batch_size=1,
        guidance_scale=2.0,
        infer_steps=20,
        sampler='dpmpp_2m_karras',
        train_steps=0,
):
    from hydit.diffusion.pipeline import StableDiffusionPipeline
    from diffusers import schedulers
    from hydit.constants import SAMPLER_FACTORY
    from hydit.modules.posemb_layers import get_fill_resize_and_crop, get_2d_rotary_pos_embed
    from hydit.modules.models import HUNYUAN_DIT_CONFIG

    import traceback
    with torch.cuda.amp.autocast():

        workspace_dir = TRAIN_CONFIG.get("workspace_dir")
        sample_config_file = TRAIN_CONFIG.get("sample_config_file", None)
        if sample_config_file is None:
            print("sample_config_file is not set.")
            return
        try:
            sample_config = json.load(open(sample_config_file, "r"))
        except Exception as e:
            print(f"Failed to load sample_config_file: {sample_config_file}")
            return
        sample_images_dir = os.path.join(workspace_dir, "sample_images")
        os.makedirs(sample_images_dir, exist_ok=True)

        sampler_factory = SAMPLER_FACTORY.copy()

        sampler_factory["uni_pc"] = {
            'scheduler': 'UniPCMultistepScheduler',
            'name': 'UniPCMultistepScheduler',
            'kwargs': {
                'beta_schedule': 'scaled_linear',
                'beta_start': 0.00085,
                'beta_end': 0.03,
                'prediction_type': 'v_prediction',
                'trained_betas': None,
                'solver_order': 2,
            }
        }
        sampler_factory["dpmpp_2m_karras"] = {
            'scheduler': 'DPMSolverMultistepScheduler',
            'name': 'DPMSolverMultistepScheduler',
            'kwargs': {
                'beta_schedule': 'scaled_linear',
                'beta_start': 0.00085,
                'beta_end': 0.03,
                'prediction_type': 'v_prediction',
                'trained_betas': None,
                'solver_order': 2,
                'algorithm_type': 'dpmsolver++',
                "use_karras_sigmas": True,
            }
        }

        # Load sampler from factory
        kwargs = sampler_factory[sampler]['kwargs']
        scheduler = sampler_factory[sampler]['scheduler']

        # Build scheduler according to the sampler.
        scheduler_class = getattr(schedulers, scheduler)
        scheduler = scheduler_class(**kwargs)

        # Set timesteps for inference steps.
        scheduler.set_timesteps(infer_steps, "cuda")

        def calc_rope(height, width):
            model_config = HUNYUAN_DIT_CONFIG["DiT-g/2"]
            patch_size = model_config['patch_size']
            head_size = model_config['hidden_size'] // model_config['num_heads']
            th = height // 8 // patch_size
            tw = width // 8 // patch_size
            base_size = 512 // 8 // patch_size
            start, stop = get_fill_resize_and_crop((th, tw), base_size)
            sub_args = [start, stop, (th, tw)]
            rope = get_2d_rotary_pos_embed(head_size, *sub_args)
            return rope

        pipeline = StableDiffusionPipeline(vae=vae,
                                           text_encoder=text_encoder,
                                           tokenizer=tokenizer,
                                           unet=model.module,
                                           scheduler=scheduler,
                                           feature_extractor=None,
                                           safety_checker=None,
                                           requires_safety_checker=False,
                                           embedder_t5=embedder_t5,
                                           )
        pipeline = pipeline.to("cuda")
        # attr _execution_device is not defined

        style = torch.as_tensor([0, 0] * batch_size, device="cuda")

        src_size_cond = (target_width, target_height)
        size_cond = list(src_size_cond) + [target_width, target_height, 0, 0]
        image_meta_size = torch.as_tensor(
            [size_cond] * 2 * batch_size, device="cuda",)

        if type(sample_config) != list:
            sample_config = [sample_config]

        for i, sample in enumerate(sample_config):
            prompt = sample.get("prompt", "")
            negative_prompt = sample.get("negative_prompt", "")
            guidance_scale = sample.get("cfg", guidance_scale)
            infer_steps = sample.get("steps", infer_steps)
            width = sample.get("width", target_width)
            height = sample.get("height", target_height)

            freqs_cis_img = calc_rope(height, width)

            try:

                samples = pipeline(
                    height=height,
                    width=width,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=batch_size,
                    guidance_scale=guidance_scale,
                    num_inference_steps=infer_steps,
                    style=style,
                    return_dict=False,
                    use_fp16=True,
                    learn_sigma=args.learn_sigma,
                    freqs_cis_img=freqs_cis_img,
                    image_meta_size=image_meta_size,
                )[0]

                pass
            except Exception as e:
                print(f"Failed to sample images: {e} ")
                # 打印堆栈信息
                traceback.print_exc()
                print(f"Failed to sample pipeline: {pipeline} ")

            # print("samples:",type(samples),)
            # input("Press Enter to continue...")
            # print("samples:",samples,)

            if type(samples) == list:
                pil_image = samples[0]
            else:
                pil_image = samples

            sample_filename = f"{args.task_flag}_train_steps_{train_steps:07d}.png"
            sample_filename_path = os.path.join(
                sample_images_dir, sample_filename)
            pil_image.save(sample_filename_path)

    return None


def model_resume(args, model, ema, logger):
    """
    Load pretrained weights.
    """
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    resume_path = UNET_PATH

    logger.info(f"Resume from checkpoint {resume_path}")

    if args.resume_split:
        # Resume main model

        resume_ckpt_module = torch.load(
            resume_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(resume_ckpt_module, strict=False)

        # Resume ema model
        if args.use_ema:
            if args.module_to_ema:
                if "resume_ckpt_module" in locals():
                    logger.info(f"    Resume ema model from main states.")
                    ema.load_state_dict(resume_ckpt_module, strict=args.strict)
                else:
                    logger.info(f"    Resume ema model from module states.")
                    resume_ckpt_module = torch.load(
                        resume_path, map_location=lambda storage, loc: storage)
                    ema.load_state_dict(resume_ckpt_module, strict=args.strict)
            else:
                if "resume_ckpt_ema" in locals():
                    logger.info(f"    Resume ema model from EMA states.")
                    ema.load_state_dict(resume_ckpt_ema, strict=args.strict)
                else:
                    logger.info(f"    Resume ema model from EMA states.")
                    resume_ckpt_ema = torch.load(resume_path,
                                                 map_location=lambda storage, loc: storage)
                    ema.load_state_dict(resume_ckpt_ema, strict=args.strict)
    else:
        raise ValueError(
            "    “If `resume` is True, then either `resume_split` must be true.”")

    return model, ema, start_epoch, start_epoch_step, train_steps


import tqdm

import requests


def set_master_port(port):
    global master_port
    master_port = port


master_port = 0


def LOG(log):
    if master_port == 0:
        raise Exception("master_port is 0")
    # 发送http
    resp = requests.request("post", f"http://127.0.0.1:{master_port}/log", data=json.dumps(log), headers={
                            "Content-Type": "application/json"})
    if resp.status_code != 200:
        raise Exception(f"LOG failed: {resp.text}")


# with tqdm(total=total_steps, initial=train_steps) as pbar:
# pbar.update(1)
# pbar.set_description(
#     f"Epoch {epoch}, step {step}, loss {loss.item():.4f}, mean_loss {mean_loss / step:.4f}")
class PBar:
    def __init__(self, total):
        self.pbar = tqdm.tqdm(total=total)

    def step(self, desc, total_steps, train_steps):
        self.pbar.update(1)
        self.pbar.set_description(desc)

        LOG({
            "type": "sample_images",
            "global_step": train_steps,
            "total_steps": total_steps,
            # "latent": noise_pred_latent_path,
        })


from torch import nn


class CustomizeEmbedsModel(nn.Module):
    dtype = torch.float16
    # x = torch.zeros(1, 1, 256, 2048)
    x = None

    def __init__(self, *args, **kwargs):
        super().__init__()

    def to(self, *args, **kwargs):
        self.dtype = torch.float16
        return self

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if self.x is None:
            if input_ids is None:
                batch_size = 1
            else:
                batch_size = input_ids.shape[0]
            self.x = torch.zeros(1, batch_size, 256, 2048, dtype=self.dtype)

        if kwargs.get("output_hidden_states", False):
            return {
                "hidden_states": self.x.to("cuda"),
                "input_ids": torch.zeros(1, 1),
            }
        return self.x


class CustomizeTokenizer(dict):

    added_tokens_encoder = []
    input_ids = torch.zeros(1, 256)
    attention_mask = torch.zeros(1, 256)

    def __init__(self, *args, **kwargs):
        self['added_tokens_encoder'] = self.added_tokens_encoder
        self['input_ids'] = self.input_ids
        self['attention_mask'] = self.attention_mask

    def tokenize(self, text):
        return text

    def __call__(self, *args, **kwargs):
        return self


class CustomizeEmbeds():
    def __init__(self):
        super().__init__()
        self.tokenizer = CustomizeTokenizer()
        self.model = CustomizeEmbedsModel().to("cuda")
        self.max_length = 256
