import os

os.system("title hook_kohya_ss_run")
import random
import time

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


import sys
sys.path.append(os.path.dirname(__file__))
try:
    import hook_kohya_ss_utils
except:
    from . import hook_kohya_ss_utils

other_config = {}
original_save_model = None


train_config = {}

sample_images_pipe_class = None


def utils_sample_images(*args, **kwargs):
    return sample_images(None, *args, **kwargs)


def get_datasets():
    import library.config_util
    user_config = library.config_util.load_user_config(
        train_config.get("dataset_config", None))
    datasets = user_config.get("datasets", [])
    if len(datasets) == 0:
        return None
    return datasets[0]


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

    controlnet = kwargs.get("controlnet", None)

    if epoch is not None and cmd_args.save_every_n_epochs is not None and epoch % cmd_args.save_every_n_epochs == 0:

        datasets = get_datasets()
        resolution = datasets.get("resolution", (512, 512))
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        height, width = resolution
        print(f"sample_images: height = {height}, width = {width}")

        prompt_dict_list = other_config.get("prompt_dict_list", [])
        if len(prompt_dict_list) == 0:
            sample_prompt = other_config.get("sample_prompt", None)
            if sample_prompt is not None:
                seed = other_config.get("seed", 0)
                prompt_dict = {
                    "controlnet_image": other_config.get("controlnet_image", None),
                    "prompt": other_config.get("sample_prompt", ""),
                    "seed": seed,
                    "negative_prompt": "",
                    "enum": 0,
                    "sample_sampler": "euler_a",
                    "sample_steps": 20,
                    "scale": 5.0,
                    "height": height,
                    "width": width,
                }
                #
                prompt_dict_list.append(prompt_dict)
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

        if prompt_dict_list is not None and len(prompt_dict_list) > 0:
            hook_kohya_ss_utils.generate_image(
                pipe_class=sample_images_pipe_class,
                cmd_args=cmd_args,
                accelerator=accelerator,
                epoch=epoch,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                vae=vae,
                prompt_dict_list=prompt_dict_list,
                controlnet=controlnet,
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
    train_args = config2args(sdxl_train_network.setup_parser(), train_config)

    LOG({
        "type": "start_train",
    })
    trainer.train(train_args)


from types import SimpleNamespace


class SimpleNamespaceCNWarrper(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)  # or self.__dict__ = kwargs
        self.__dict__["mid_block_type"] = "UNetMidBlock2DCrossAttn"
        self.__dict__["_diffusers_version"] = "0.6.0"
        self.__iter__ = lambda: iter(kwargs.keys())
    # is not iterable

    def __iter__(self):
        return iter(self.__dict__.keys())
    # object has no attribute 'num_attention_heads'

    def __getattr__(self, name):
        return self.__dict__.get(name, None)


func_map = {
    "run_lora_sd1_5": run_lora_sd1_5,
    "run_lora_sdxl": run_lora_sdxl,
}


import requests


def LOG(log):
    try:
        # 发送http
        resp = requests.request("post", f"http://127.0.0.1:{master_port}/log", data=json.dumps(log), headers={
                                "Content-Type": "application/json"})
        if resp.status_code != 200:
            # raise Exception(f"LOG failed: {resp.text}")
            print(f"LOG failed: {resp.text}")
    except Exception as e:
        print(f"LOG failed: {e}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sys_path", type=str, default="")
        parser.add_argument("--config", type=str, default="")
        parser.add_argument("--train_func", type=str, default="")
        parser.add_argument("--master_port", type=int, default=0)
        args = parser.parse_args()

        master_port = args.master_port

        print(f"master_port = {master_port}")

        sys_path = args.sys_path
        if sys_path != "":
            sys.path.append(sys_path)

        config_file = args.config
        if config_file == "":
            raise Exception("train_config is empty")

        global_config = {}
        with open(config_file, "r") as f:
            _global_config = f.read()
            global_config = json.loads(_global_config)

        train_config = global_config.get("train_config")
        print(f"""=======================train_config=======================
    {json.dumps(train_config, indent=4, ensure_ascii=False)}
            """)

        other_config = global_config.get("other_config", {})
        print(f"""=======================other_config=======================
    {json.dumps(other_config, indent=4, ensure_ascii=False)}
            """)

        train_func = args.train_func
        if train_func == "":
            raise Exception("train_func is empty")

        print(f"train_func = {train_func}")

        time.sleep(2)
        LOG({
            "type": "Read configuration completed!",
        })

        func_map[train_func]()
    except Exception as e:
        print(f"Exception: {e}")
        if sys.platform == "win32":
            input("Press Enter to continue...")
