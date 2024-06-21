import os
import random

import torch
import logging
import sys
import json
import importlib
import argparse
import toml


def config2args(train_parser: argparse.ArgumentParser, config):

    config_args_list = []
    for key, value in config.items():
        if type(value) == bool:
            if value:
                config_args_list.append(f"--{key}")
        else:
            config_args_list.append(f"--{key}")
            config_args_list.append(str(value))
    args = train_parser.parse_args(config_args_list)
    return args


from PIL import Image


import numpy as np
import tempfile
import safetensors.torch

import hook_kohya_ss_utils


other_config = {}
original_save_model = None


train_config_json = "{}"

sample_images_pipe_class = None


def sample_images(self, *args, **kwargs):
    #  accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet
    accelerator = args[0]
    cmd_args = args[1]
    epoch = args[2]
    global_step = args[3]
    device = args[4]
    vae = args[5]
    tokenizer = args[6]
    text_encoder = args[7]
    unet = args[8]

    # print(f"sample_images: args = {args}")
    # print(f"sample_images: kwargs = {kwargs}")

    # last_noise_pred = hook_kohya_ss_utils.running_info.get(
    #     "last_noise_pred", None)

    # noise_pred_latent_path = None
    # if last_noise_pred is not None:
    #     noise_pred_latent_path = os.path.join(
    #         tempfile.gettempdir(), f"sample_images_{global_step % 5}.latent")

    #     output = {}
    #     output["latent_tensor"] = last_noise_pred
    #     output["latent_format_version_0"] = torch.tensor([])
    #     safetensors.torch.save_file(output, noise_pred_latent_path)

    if epoch is not None and cmd_args.save_every_n_epochs is not None and epoch % cmd_args.save_every_n_epochs == 0:

        prompt_dict_list = other_config.get("prompt_dict_list", [])
        if len(prompt_dict_list) == 0:
            seed = other_config.get("seed", 0)

            prompt_dict_list.append({
                "controlnet_image": None,
                "prompt": other_config.get("sample_prompt", ""),
                "seed": seed,
                "negative_prompt": "",
                "enum": 0,
                "sample_sampler": "euler_a",
                "sample_steps": 20,
                "scale": 5.0,
            })
        else:
            for i, prompt_dict in enumerate(prompt_dict_list):
                if prompt_dict.get("controlnet_image", None) is None:
                    prompt_dict["controlnet_image"] = None
                if prompt_dict.get("seed", None) is None:
                    prompt_dict["seed"] = 0
                if prompt_dict.get("negative_prompt", None) is None:
                    prompt_dict["negative_prompt"] = ""
                if prompt_dict.get("enum", None) is None:
                    prompt_dict["enum"] = i
        hook_kohya_ss_utils.generate_image(
            cmd_args=cmd_args,
            accelerator=accelerator,
            epoch=epoch,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            pipe_class=sample_images_pipe_class,
            prompt_dict_list=prompt_dict_list,
        )

    LOG({
        "type": "sample_images",
        "global_step": global_step,
        "total_steps": cmd_args.max_train_steps,
        # "latent": noise_pred_latent_path,
    })


def run_lora_sd1_5():
    hook_kohya_ss_utils.hook_kohya_ss()

    # 覆盖sample_images生成函数,包括进度条和生成图片功能
    import train_network
    train_network.NetworkTrainer.sample_images = sample_images

    # 配置对应的pipeline
    import library.train_util
    global sample_images_pipe_class
    sample_images_pipe_class = library.train_util.StableDiffusionLongPromptWeightingPipeline

    trainer = train_network.NetworkTrainer()
    train_config = json.loads(train_config_json)
    train_args = config2args(train_network.setup_parser(), train_config)

    LOG({
        "type": "start_train",
    })
    trainer.train(train_args)


def run_lora_sdxl():
    hook_kohya_ss_utils.hook_kohya_ss()

    # 覆盖sample_images生成函数,包括进度条和生成图片功能
    import sdxl_train_network
    sdxl_train_network.SdxlNetworkTrainer.sample_images = sample_images

    # 配置对应的pipeline
    import library.sdxl_train_util
    global sample_images_pipe_class
    sample_images_pipe_class = library.sdxl_train_util.SdxlStableDiffusionLongPromptWeightingPipeline

    trainer = sdxl_train_network.SdxlNetworkTrainer()
    train_config = json.loads(train_config_json)
    train_args = config2args(sdxl_train_network.setup_parser(), train_config)

    LOG({
        "type": "start_train",
    })
    trainer.train(train_args)


func_map = {
    "run_lora_sd1_5": run_lora_sd1_5,
    "run_lora_sdxl": run_lora_sdxl,
}


import requests


def LOG(log):
    # 发送http
    resp = requests.request("post", f"http://127.0.0.1:{master_port}/log", data=json.dumps(log), headers={
                            "Content-Type": "application/json"})
    if resp.status_code != 200:
        raise Exception(f"LOG failed: {resp.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_path", type=str, default="")
    parser.add_argument("--train_config_json", type=str, default="")
    parser.add_argument("--train_func", type=str, default="")
    parser.add_argument("--master_port", type=int, default=0)
    parser.add_argument("--other_config_json", type=str, default="{}")
    args = parser.parse_args()

    other_config_json = args.other_config_json
    other_config = json.loads(other_config_json)

    master_port = args.master_port

    print(f"master_port = {master_port}")

    sys_path = args.sys_path
    if sys_path != "":
        sys.path.append(sys_path)

    train_config_json = args.train_config_json

    train_func = args.train_func
    if train_func == "":
        raise Exception("train_func is empty")

    func_map[train_func]()
