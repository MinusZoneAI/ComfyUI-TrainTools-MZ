import os

import torch
os.environ["HTTP_PROXY"] = "http://127.0.0.1:0"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:0"
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


def sample_images(self, *args, **kwargs):
    #  accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet
    global_step = args[3]
    last_noise_pred = hook_kohya_ss_utils.running_info.get(
        "last_noise_pred", None)
    noise_pred_latent_path = None
    if last_noise_pred is not None:
        noise_pred_latent_path = os.path.join(
            tempfile.gettempdir(), f"sample_images_{global_step % 5}.latent")

        output = {}
        output["latent_tensor"] = last_noise_pred
        output["latent_format_version_0"] = torch.tensor([])
        safetensors.torch.save_file(output, noise_pred_latent_path)

    LOG(json.dumps({
        "type": "sample_images",
        "global_step": global_step,
        "latent": noise_pred_latent_path,
    }))


def run_lora_sd1_5():
    hook_kohya_ss_utils.hook_kohya_ss()
    import train_network
    train_network.NetworkTrainer.sample_images = sample_images
    trainer = train_network.NetworkTrainer()
    train_config = json.loads(train_config_json)
    train_args = config2args(train_network.setup_parser(), train_config)

    trainer.train(train_args)


def run_lora_sdxl():
    hook_kohya_ss_utils.hook_kohya_ss()
    import hook_kohya_ss_utils
    import sdxl_train_network
    sdxl_train_network.SdxlNetworkTrainer.sample_images = sample_images
    trainer = sdxl_train_network.SdxlNetworkTrainer()
    train_config = json.loads(train_config_json)
    train_args = config2args(sdxl_train_network.setup_parser(), train_config)
    trainer.train(train_args)


func_map = {
    "run_lora_sd1_5": run_lora_sd1_5,
    "run_lora_sdxl": run_lora_sdxl,
}


import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_port = 0


def is_connected():
    try:
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    except socket.error:
        return False
    return True


def LOG(log):
    sock.sendto(log.encode(), ('127.0.0.1', udp_port))
    if not is_connected():
        raise Exception("Connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_path", type=str, default="")
    parser.add_argument("--train_config_json", type=str, default="")
    parser.add_argument("--train_func", type=str, default="")
    parser.add_argument("--udp_port", type=int, default=0)
    args = parser.parse_args()

    udp_port = args.udp_port

    print(f"udp_port = {udp_port}")

    sys_path = args.sys_path
    if sys_path != "":
        sys.path.append(sys_path)

    train_config_json = args.train_config_json

    train_func = args.train_func
    if train_func == "":
        raise Exception("train_func is empty")

    func_map[train_func]()
