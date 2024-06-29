
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch


class SimpleNamespaceCNWarrper(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)
        self.__iter__ = lambda: iter(kwargs.keys())
    # is not iterable

    def __iter__(self):
        return iter(self.__dict__.keys())
    # object has no attribute 'num_attention_heads'

    def __getattr__(self, name):
        return self.__dict__.get(name, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
    )
    parser.add_argument("--sys_path", type=str, default="")
    parser.add_argument("--train_config_file", type=str, default="")
    parser.add_argument("--mz_master_port", type=int, default=0)
    args = parser.parse_args()

    master_port = args.mz_master_port

    print(f"master_port = {master_port}")

    try:
        from . import hook_HYDiT_utils
    except Exception as e:
        import hook_HYDiT_utils

    hook_HYDiT_utils.set_master_port(master_port)

    sys_path = args.sys_path
    if sys_path != "":
        sys.path.append(sys_path)

    print("HYDi run hook")

    try:
        from . import hook_HYDiT_main_train_deepspeed
    except Exception as e:
        import hook_HYDiT_main_train_deepspeed

    import hydit.config

    def _handle_conflict_error(self, *args, **kwargs):
        pass

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        return args

    train_config_file = args.train_config_file

    if train_config_file == "":
        raise ValueError("train_config_file is empty")

    train_config = {}
    with open(train_config_file, "r") as f:
        train_config = json.load(f)

    argparse.ArgumentParser._handle_conflict_error = _handle_conflict_error
    argparse.ArgumentParser._handle_conflict_resolve = _handle_conflict_error
    argparse.ArgumentParser.parse_args = parse_args
    margs = hydit.config.get_args()
    margs.model = train_config.get("model", "DiT-g/2")

    margs.task_flag = train_config.get(
        "task_flag", "lora_porcelain_ema_rank64")

    margs.resume_split = train_config.get("resume_split", True)
    margs.ema_to_module = train_config.get("ema_to_module", True)
    margs.deepspeed = False
    margs.predict_type = train_config.get("predict_type", "v_prediction")
    margs.training_parts = train_config.get("training_parts", "lora")
    margs.batch_size = train_config.get("batch_size", 1)
    margs.grad_accu_steps = train_config.get("grad_accu_steps", 1)
    margs.global_seed = train_config.get("global_seed", 0)
    margs.use_flash_attn = train_config.get("use_flash_attn", False)
    margs.use_fp16 = train_config.get("use_fp16", True)
    margs.qk_norm = train_config.get("qk_norm", True)
    margs.ema_dtype = train_config.get("ema_dtype", "fp32")
    margs.async_ema = False
    margs.multireso = train_config.get("multireso", True)
    margs.epochs = train_config.get("epochs", 50)
    margs.target_ratios = train_config.get(
        "target_ratios", ['1:1', '3:4', '4:3', '16:9', '9:16'])
    margs.rope_img = train_config.get("rope_img", "base1024")

    margs.image_size = train_config.get("image_size", 1024)

    margs.rope_real = train_config.get("rope_real", True)

    margs.index_file = train_config.get("index_file", None)

    margs.lr = train_config.get("lr", 1e-5)
    
    margs.rank = train_config.get("rank", 8)

    margs.noise_offset = train_config.get("noise_offset", 0.0)

    margs.log_every = train_config.get("log_every", 99999999999999)
    
    margs.use_zero_stage = train_config.get("use_zero_stage", 2)

    margs.global_batch_size = train_config.get("global_batch_size", 1)
    margs.deepspeed = True

    margs.results_dir = train_config.get("results_dir")

    margs.mse_loss_weight_type = train_config.get(
        "mse_loss_weight_type", "constant")

    for k, v in train_config.items():
        if hasattr(margs, k):
            setattr(margs, k, v)

    hook_HYDiT_utils.set_unet_path(
        train_config.get("unet_path"))

    hook_HYDiT_utils.set_vae_ema_path(
        train_config.get("vae_ema_path"))

    hook_HYDiT_utils.set_text_encoder_path(
        train_config.get("text_encoder_path"))

    hook_HYDiT_utils.set_tokenizer_path(
        train_config.get("tokenizer_path"))

    hook_HYDiT_utils.set_t5_encoder_path(
        train_config.get("t5_encoder_path"))

    hook_HYDiT_utils.set_train_config(train_config)

    try:
        # deepspeed/runtime/engine
        import deepspeed.runtime.engine
        deepspeed.runtime.engine.DeepSpeedEngine._do_sanity_check = lambda x: None
    except Exception as e:
        pass

    if type(margs.image_size) == int:
        margs.image_size = [margs.image_size, margs.image_size]
    hook_HYDiT_main_train_deepspeed.Core(margs)
