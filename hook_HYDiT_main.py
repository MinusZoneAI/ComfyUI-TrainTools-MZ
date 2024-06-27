
import argparse
import json
import logging
import os
import sys
from types import SimpleNamespace
import gc
import json
import os
import random
import sys
import time
from functools import partial
from glob import glob
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from diffusers.models import AutoencoderKL
from transformers import BertModel, BertTokenizer, logging as tf_logging

# from hydit.constants import VAE_EMA_PATH, TEXT_ENCODER, TOKENIZER, T5_ENCODER
from hydit.lr_scheduler import WarmupLR
from hydit.data_loader.arrow_load_stream import TextImageArrowStream
from hydit.diffusion import create_diffusion
from hydit.modules.fp16_layers import Float16Module
from hydit.modules.models import HUNYUAN_DIT_MODELS
from hydit.modules.posemb_layers import init_image_posemb
from hydit.utils.tools import set_seeds, get_trainable_params
from IndexKits.index_kits import ResolutionGroup
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


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
    'MT5': 'ckpts/t2i/mt5',
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

def create_logger(log_dir=None):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    return logger


def create_exp_folder(args, rank):
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

    # print('args.results_dir', args.results_dir)
    _existed_experiments = os.listdir(args.results_dir)
    # print('_existed_experiments', _existed_experiments)
    existed_experiments = []
    for x in _existed_experiments:
        if os.path.isdir(os.path.join(args.results_dir, x)):
            if x.split('-')[0].isdigit():
                existed_experiments.append(x)

    if len(existed_experiments) == 0:
        experiment_index = 1
    else:
        existed_experiments.sort()
        # print('existed_experiments', existed_experiments)
        experiment_index = max([int(x.split('-')[0])
                               for x in existed_experiments]) + 1
    model_string_name = args.task_flag if args.task_flag else args.model.replace(
        "/", "-")
    # Create an experiment folder
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    # Stores saved model checkpoints
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger()
        experiment_dir = ""
    logger.info = print
    return experiment_dir, checkpoint_dir, logger


def save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir):
    def save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.pt"):
        cur_ckpt_save_dir = f"{checkpoint_dir}/{tag}"
        if rank == 0:
            if args.use_fp16:
                model.module.save_pretrained(cur_ckpt_save_dir)
            else:
                model.save_pretrained(cur_ckpt_save_dir)

    checkpoint_path = "[Not rank 0. Disabled output.]"

    client_state = {
        "steps": train_steps,
        "epoch": epoch,
        "args": args
    }
    if ema is not None:
        client_state['ema'] = ema.state_dict()

    dst_paths = []

    if train_steps % args.ckpt_every == 0:
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state,
                                 tag=f"{train_steps:07d}.pt")
            else:
                model.save_checkpoint(
                    checkpoint_dir, client_state=client_state, tag=f"{train_steps:07d}.pt")
            dst_paths.append(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except:
            logger.error(f"Saved failed to {checkpoint_path}")

    if train_steps % args.ckpt_latest_every == 0 or train_steps == args.max_training_steps:
        save_name = "latest.pt"
        checkpoint_path = f"{checkpoint_dir}/{save_name}"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state,
                                 tag=f"{save_name}")
            else:
                model.save_checkpoint(
                    checkpoint_dir, client_state=client_state, tag=f"{save_name}")
            dst_paths.append(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except:
            logger.error(f"Saved failed to {checkpoint_path}")

    return checkpoint_path


@torch.no_grad()
def prepare_model_inputs(args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img):
    image, text_embedding, text_embedding_mask, text_embedding_t5, text_embedding_mask_t5, kwargs = batch

    # clip & mT5 text embedding
    text_embedding = text_embedding.to(device)
    text_embedding_mask = text_embedding_mask.to(device)
    encoder_hidden_states = text_encoder(
        text_embedding.to(device),
        attention_mask=text_embedding_mask.to(device),
    )[0]
    text_embedding_t5 = text_embedding_t5.to(device).squeeze(1)
    text_embedding_mask_t5 = text_embedding_mask_t5.to(device).squeeze(1)
    with torch.no_grad():
        output_t5 = text_encoder_t5(
            input_ids=text_embedding_t5,
            attention_mask=text_embedding_mask_t5 if T5_ENCODER['attention_mask'] else None,
            output_hidden_states=True
        )
        encoder_hidden_states_t5 = output_t5['hidden_states'][T5_ENCODER['layer_index']].detach(
        )

    # additional condition
    image_meta_size = kwargs['image_meta_size'].to(device)
    style = kwargs['style'].to(device)

    if args.extra_fp16:
        image = image.half()
        image_meta_size = image_meta_size.half() if image_meta_size is not None else None

    # Map input images to latent space + normalize latents:
    image = image.to(device)
    vae_scaling_factor = vae.config.scaling_factor
    latents = vae.encode(image).latent_dist.sample().mul_(
        vae_scaling_factor).to(device)

    # positional embedding
    _, _, height, width = image.shape
    reso = f"{height}x{width}"

    # print("freqs_cis_img:", freqs_cis_img.keys())
    # print("freqs_cis_img[reso]:", freqs_cis_img[reso])

    cos_cis_img, sin_cis_img = freqs_cis_img[reso]

    # Model conditions
    model_kwargs = dict(
        encoder_hidden_states=encoder_hidden_states,
        text_embedding_mask=text_embedding_mask,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        text_embedding_mask_t5=text_embedding_mask_t5,
        image_meta_size=image_meta_size,
        style=style,
        cos_cis_img=cos_cis_img,
        sin_cis_img=sin_cis_img,
    )

    return latents, model_kwargs


def Core(args, LOG):
    if args.training_parts == "lora":
        args.use_ema = False

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    batch_size = args.batch_size
    grad_accu_steps = args.grad_accu_steps
    global_batch_size = batch_size * grad_accu_steps

    seed = args.global_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Starting  ,seed={seed}.")
    rank = 0
    # Setup an experiment folder
    experiment_dir, checkpoint_dir, logger = create_exp_folder(args, rank)

    # Log all the arguments
    logger.info(sys.argv)
    logger.info(str(args))
    # Save to a json file
    args_dict = vars(args)
    with open(f"{experiment_dir}/args.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Disable the message "Some weights of the model checkpoint at ... were not used when initializing BertModel."
    # If needed, just comment the following line.
    tf_logging.set_verbosity_error()

    # ===========================================================================
    # Building HYDIT
    # ===========================================================================

    logger.info("Building HYDIT Model.")

    # ---------------------------------------------------------------------------
    #   Training sample base size, such as 256/512/1024. Notice that this size is
    #   just a base size, not necessary the actual size of training samples. Actual
    #   size of the training samples are correlated with `resolutions` when enabling
    #   multi-resolution training.
    # ---------------------------------------------------------------------------
    image_size = args.image_size
    if type(image_size) == int:
        image_size = [image_size, image_size]
    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
    if len(image_size) != 2:
        raise ValueError(f"Invalid image size: {args.image_size}")
    assert image_size[0] % 8 == 0 and image_size[1] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder). " \
                                                              f"got {image_size}"
    latent_size = [image_size[0] // 8, image_size[1] // 8]

    # initialize model by deepspeed
    model = HUNYUAN_DIT_MODELS[args.model](args,
                                           input_size=latent_size,
                                           log_fn=logger.info,
                                           )
    # Multi-resolution / Single-resolution training.
    if args.multireso:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      step=args.reso_step,
                                      target_ratios=args.target_ratios).data
    else:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      target_ratios=['1:1']).data

    freqs_cis_img = init_image_posemb(args.rope_img,
                                      resolutions=resolutions,
                                      patch_size=model.patch_size,
                                      hidden_size=model.hidden_size,
                                      num_heads=model.num_heads,
                                      log_fn=logger.info,
                                      rope_real=args.rope_real,
                                      )

    # Create EMA model and convert to fp16 if needed.
    ema = None
    device = "cuda"
    if args.use_ema:
        raise ValueError("Not support EMA model.")

    # Setup FP16 main model:
    if args.use_fp16:
        model = Float16Module(model, args)
    logger.info(
        f"    Using main model with data type {'fp16' if args.use_fp16 else 'fp32'}")

    diffusion = create_diffusion(
        noise_schedule=args.noise_schedule,
        predict_type=args.predict_type,
        learn_sigma=args.learn_sigma,
        mse_loss_weight_type=args.mse_loss_weight_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_offset=args.noise_offset,
    )

    # Setup VAE
    logger.info(f"    Loading vae from {VAE_EMA_PATH}")

    if os.path.isdir(VAE_EMA_PATH):
        vae = AutoencoderKL.from_pretrained(VAE_EMA_PATH)
    else:
        vae = AutoencoderKL.from_single_file(VAE_EMA_PATH)
    # Setup BERT text encoder
    logger.info(f"    Loading Bert text encoder from {TEXT_ENCODER}")
    text_encoder = BertModel.from_pretrained(
        TEXT_ENCODER, False, revision=None)
    # Setup BERT tokenizer:
    logger.info(f"    Loading Bert tokenizer from {TOKENIZER}")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
    # Setup T5 text encoder
    from hydit.modules.text_encoder import MT5Embedder
    mt5_path = T5_ENCODER['MT5']
    embedder_t5 = MT5Embedder(
        mt5_path, torch_dtype=T5_ENCODER['torch_dtype'], max_length=args.text_len_t5)
    tokenizer_t5 = embedder_t5.tokenizer
    text_encoder_t5 = embedder_t5.model

    if args.extra_fp16:
        logger.info(f"    Using fp16 for extra modules: vae, text_encoder")
        vae = vae.half().to(device)
        text_encoder = text_encoder.half().to(device)
        text_encoder_t5 = text_encoder_t5.half().to(device)
    else:
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        text_encoder_t5 = text_encoder_t5.to(device)

    logger.info(
        f"    Optimizer parameters: lr={args.lr}, weight_decay={args.weight_decay}")

    # ===========================================================================
    # Building Dataset
    # ===========================================================================

    logger.info(f"Building Streaming Dataset.")
    logger.info(f"    Loading index file {args.index_file} (v2)")
    world_size = 1

    dataset = TextImageArrowStream(args=args,
                                   resolution=image_size[0],
                                   random_flip=args.random_flip,
                                   log_fn=logger.info,
                                   index_file=args.index_file,
                                   multireso=args.multireso,
                                   batch_size=batch_size,
                                   world_size=world_size,
                                   random_shrink_size_cond=args.random_shrink_size_cond,
                                   merge_src_cond=args.merge_src_cond,
                                   uncond_p=args.uncond_p,
                                   text_ctx_len=args.text_len,
                                   tokenizer=tokenizer,
                                   uncond_p_t5=args.uncond_p_t5,
                                   text_ctx_len_t5=args.text_len_t5,
                                   tokenizer_t5=tokenizer_t5,
                                   )
    if args.multireso:
        sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
                                          shuffle=False, drop_last=True, batch_size=batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
                                                   shuffle=False, drop_last=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"    Dataset contains {len(dataset):,} images.")
    logger.info(f"    Index file: {args.index_file}.")
    if args.multireso:
        logger.info(f'    Using MultiResolutionBucketIndexV2 with step {dataset.index_manager.step} '
                    f'and base size {dataset.index_manager.base_size}')
        logger.info(f'\n  {dataset.index_manager.resolutions}')

    # ===========================================================================
    # Loading parameter
    # ===========================================================================

    logger.info(f"Loading parameter")
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    # Resume checkpoint if needed
    model, ema, start_epoch, start_epoch_step, train_steps = model_resume(
        args, model, ema, logger)

    if args.training_parts == "lora":
        loraconfig = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            target_modules=args.target_modules
        )
        if args.use_fp16:
            model.module = get_peft_model(model.module, loraconfig)
        else:
            model = get_peft_model(model, loraconfig)

    logger.info(f"    Training parts: {args.training_parts}")

    # ===========================================================================
    # Training
    # ===========================================================================

    model.to(device)
    model.train()
    if args.use_ema:
        ema.eval()

    print(f"    Worker {rank} ready.")

    iters_per_epoch = len(loader)
    logger.info(
        " ****************************** Running training ******************************")
    logger.info(f"      Number GPUs:               {world_size}")
    logger.info(f"      Number training samples:   {len(dataset):,}")
    logger.info(
        f"      Number parameters:         {sum(p.numel() for p in model.parameters()):,}")
    logger.info(
        f"      Number trainable params:   {sum(p.numel() for p in get_trainable_params(model)):,}")
    logger.info(
        "    ------------------------------------------------------------------------------")
    logger.info(f"      Iters per epoch:           {iters_per_epoch:,}")
    logger.info(f"      Batch size per device:     {batch_size}")
    logger.info(
        f"      Batch size all device:     {batch_size * world_size * grad_accu_steps:,} (world_size * batch_size * grad_accu_steps)")
    logger.info(f"      Gradient Accu steps:       {args.grad_accu_steps}")
    logger.info(
        f"      Total optimization steps:  {args.epochs * iters_per_epoch // grad_accu_steps:,}")

    logger.info(
        f"      Training epochs:           {start_epoch}/{args.epochs}")
    logger.info(
        f"      Training epoch steps:      {start_epoch_step:,}/{iters_per_epoch:,}")
    logger.info(
        f"      Training total steps:      {train_steps:,}/{min(args.max_training_steps, args.epochs * iters_per_epoch):,}")
    logger.info(
        "    ------------------------------------------------------------------------------")
    logger.info(f"      Noise schedule:            {args.noise_schedule}")
    logger.info(
        f"      Beta limits:               ({args.beta_start}, {args.beta_end})")
    logger.info(f"      Learn sigma:               {args.learn_sigma}")
    logger.info(f"      Prediction type:           {args.predict_type}")
    logger.info(f"      Noise offset:              {args.noise_offset}")

    logger.info(
        "    ------------------------------------------------------------------------------")
    logger.info(
        f"      Using EMA model:           {args.use_ema} ({args.ema_dtype})")
    if args.use_ema:
        logger.info(
            f"      Using EMA decay:           {ema.max_value if args.use_ema else None}")
        logger.info(
            f"      Using EMA warmup power:    {ema.power if args.use_ema else None}")
    logger.info(f"      Using main model fp16:     {args.use_fp16}")
    logger.info(f"      Using extra modules fp16:  {args.extra_fp16}")
    logger.info(
        "    ------------------------------------------------------------------------------")
    logger.info(f"      Experiment directory:      {experiment_dir}")
    logger.info(
        "    *******************************************************************************")

    if args.gc_interval > 0:
        gc.disable()
        gc.collect()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    if args.async_ema:
        ema_stream = torch.cuda.Stream()

    total_steps = args.epochs * len(loader)

    if args.use_fp16:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=TRAIN_CONFIG.get("adam_epsilon", 1e-7))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    easy_sample_images(args, vae, text_encoder, tokenizer, model, embedder_t5,
                       target_height=768, target_width=1280, train_steps=0)
    # torch.autograd.set_detect_anomaly(True)
    # with torch.autograd.detect_anomaly():
    with tqdm(total=total_steps, initial=train_steps) as pbar:

        # Training loop
        for epoch in range(start_epoch, args.epochs):
            logger.info(f"    Start random shuffle with seed={seed}")
            # Makesure all processors use the same seed to shuffle dataset.
            dataset.shuffle(seed=args.global_seed + epoch, fast=True)
            logger.info(f"    End of random shuffle")

            # Move sampler to start_index
            if not args.multireso:
                start_index = start_epoch_step * world_size * batch_size
                if start_index != sampler.start_index:
                    sampler.start_index = start_index
                    # Reset start_epoch_step to zero, to ensure next epoch will start from the beginning.
                    start_epoch_step = 0
                    logger.info(
                        f"      Iters left this epoch: {len(loader):,}")

            logger.info(f"    Beginning epoch {epoch}...")
            step = 0
            mean_loss = 0
            for batch in loader:
                step += 1

                # Zero gradients, backward pass, and optimizer step:
                optimizer.zero_grad()
                latents, model_kwargs = prepare_model_inputs(
                    args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img)
                # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):

                if True:
                    # training model by deepspeed while use fp16
                    if args.use_fp16:
                        if args.use_ema and args.async_ema:
                            with torch.cuda.stream(ema_stream):
                                ema.update(model.module.module, step=step)
                            torch.cuda.current_stream().wait_stream(ema_stream)

                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                        loss_dict = diffusion.training_losses(
                            model=model, x_start=latents, model_kwargs=model_kwargs)

                        loss = loss_dict["loss"].mean()

                    # print(f"step={step}, loss={loss.item()}")

                # model.backward(loss)

                loss.backward()

                # last_batch_iteration = (
                #     train_steps + 1) // (global_batch_size // (batch_size * world_size))
                # model.step(
                #     lr_kwargs={'last_batch_iteration': last_batch_iteration})

                optimizer.step()

                if args.use_ema and not args.async_ema or (args.async_ema and step == len(loader) - 1):
                    if args.use_fp16:
                        ema.update(model.module.module, step=step)
                    else:
                        ema.update(model.module, step=step)

                train_steps += 1
                mean_loss += loss.item()
                pbar.update(1)
                pbar.set_description(
                    f"Epoch {epoch}, step {step}, loss {loss.item():.4f}, mean_loss {mean_loss / step:.4f}")

                # ===========================================================================
                # Log loss values:
                # ===========================================================================
                # running_loss += loss.item()
                # log_steps += 1
                # if train_steps % args.log_every == 0:
                #     # Measure training speed:
                #     torch.cuda.synchronize()
                #     end_time = time.time()
                #     steps_per_sec = log_steps / (end_time - start_time)
                #     # Reduce loss history over all processes:
                #     avg_loss = torch.tensor(
                #         running_loss / log_steps, device=device)
                #     # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)

                #     # avg_loss = avg_loss.item() / world_size
                #     # get lr from deepspeed fused optimizer
                #     logger.info(f"(step={train_steps:07d}) " +
                #                 (f"(update_step={train_steps // args.grad_accu_steps:07d}) " if args.grad_accu_steps > 1 else "") +
                #                 f"Train Loss: {avg_loss:.4f}, "
                #                 # f"Lr: {opt.param_groups[0]['lr']:.6g}, "
                #                 f"Steps/Sec: {steps_per_sec:.2f}, "
                #                 f"Samples/Sec: {int(steps_per_sec * batch_size * world_size):d}")
                #     # Reset monitoring variables:
                #     running_loss = 0
                #     log_steps = 0
                #     start_time = time.time()

                # collect gc:
                if args.gc_interval > 0 and (step % args.gc_interval == 0):
                    gc.collect()

                if (train_steps % args.ckpt_every == 0 or train_steps % args.ckpt_latest_every == 0  # or train_steps == args.max_training_steps
                    ) and train_steps > 0:
                    logger.info(
                        f"    Saving checkpoint at step {train_steps}.")
                    easy_sample_images(args, vae, text_encoder, tokenizer, model, embedder_t5,
                                       target_height=768, target_width=1280, train_steps=train_steps)
                    save_checkpoint(args, rank, logger, model, ema,
                                    epoch, train_steps, checkpoint_dir)

                if train_steps >= args.max_training_steps:
                    logger.info(f"Breaking step loop at {train_steps}.")
                    break

                LOG({
                    "type": "sample_images",
                    "global_step": train_steps,
                    "total_steps": total_steps,
                    # "latent": noise_pred_latent_path,
                })

            if train_steps >= args.max_training_steps:
                logger.info(f"Breaking epoch loop at {epoch}.")
                break


def model_resume(args, model, ema, logger):
    """
    Load pretrained weights.
    """
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    resume_path = UNET_PATH
    if not Path(resume_path).exists():
        raise FileNotFoundError(
            f"    Cannot find checkpoint from {resume_path}")

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


from hydit.diffusion.pipeline import StableDiffusionPipeline
from diffusers import schedulers
from hydit.constants import SAMPLER_FACTORY
from hydit.modules.posemb_layers import get_fill_resize_and_crop, get_2d_rotary_pos_embed
from hydit.modules.models import HUNYUAN_DIT_CONFIG


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
        guidance_scale=5.0,
        infer_steps=20,
        sampler='ddpm',
        train_steps=0,
):
    with torch.no_grad():

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

        # Load sampler from factory
        kwargs = SAMPLER_FACTORY[sampler]['kwargs']
        scheduler = SAMPLER_FACTORY[sampler]['scheduler']

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
                                           unet=model,
                                           scheduler=scheduler,
                                           feature_extractor=None,
                                           safety_checker=None,
                                           requires_safety_checker=False,
                                           embedder_t5=embedder_t5,
                                           ).to("cuda")

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
            except Exception as e:
                print(f"Failed to sample images: {e}")
                continue

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
