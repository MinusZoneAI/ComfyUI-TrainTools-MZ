

import argparse
import hashlib
import os
import shutil
import sys
import json
import subprocess
import time

import torch

from .mz_train_tools_utils import Utils
import folder_paths
import nodes


# 初始化工具仓库和工作区
def MZ_KohyaSSInitWorkspace_call(args={}):
    mz_dir = Utils.get_minus_zone_models_path()
    git_url = "https://github.com/kohya-ss/sd-scripts"
    kohya_ss_lora_dir = os.path.join(mz_dir, "train_tools", "kohya_ss_lora")
    try:
        if not os.path.exists(kohya_ss_lora_dir) or not os.path.exists(os.path.join(kohya_ss_lora_dir, ".git")):
            subprocess.run(
                ["git", "clone", "--depth", "1", git_url, kohya_ss_lora_dir], check=True)

        # 切换远程分支 git remote set-branches origin 'main'
        branch = args.get("branch", "main")

        # 查看本地分支是否一致
        result = subprocess.run(
            ["git", "branch"], cwd=kohya_ss_lora_dir, stdout=subprocess.PIPE, check=True)
        if branch.encode() not in result.stdout:
            subprocess.run(
                ["git", "remote", "set-branches", "origin", branch], cwd=kohya_ss_lora_dir, check=True)
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", branch], cwd=kohya_ss_lora_dir, check=True)

            # 恢复所有文件
            subprocess.run(
                ["git", "checkout", "."], cwd=kohya_ss_lora_dir, check=True)

            subprocess.run(
                ["git", "checkout", branch], cwd=kohya_ss_lora_dir, check=True)

    except Exception as e:
        raise Exception(f"克隆kohya-ss/sd-scripts或者切换分支时出现异常,详细信息请查看控制台...")

    workspace_name = args.get("workspace_name", None)

    if workspace_name is None or workspace_name == "":
        raise Exception("工作区名称不能为空(workspace_name is required)")

    workspaces_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces")

    os.makedirs(workspaces_dir, exist_ok=True)

    workspace_dir = os.path.join(workspaces_dir, workspace_name)
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)

    return (
        args,
    )


def MZ_ImageSelecter_call(args={}):
    images = args.get("images")
    pil_images = Utils.tensors2pil_list(images)

    resolution = args.get("resolution", 512)

    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)

    if workspace_name is None or workspace_name == "":
        raise Exception("工作区名称不能为空(workspace_name is required)")

    workspace_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces", workspace_name)
    train_images_dir = os.path.join(workspace_dir, "train_images")
    os.makedirs(train_images_dir, exist_ok=True)

    for i, pil_image in enumerate(pil_images):
        pil_image = Utils.resize_max(pil_image, resolution, resolution)
        filename = hashlib.md5(pil_image.tobytes()).hexdigest() + ".png"
        pil_image.save(os.path.join(train_images_dir, filename))

    dataset_config_path = os.path.join(workspace_dir, "dataset.toml")
    generate_toml_config(
        dataset_config_path,
        enable_bucket=args.get("enable_bucket") == "enable",
        resolution=args.get("resolution"),
        batch_size=args.get("batch_size"),
        image_dir=train_images_dir,
        caption_extension=".caption",
        num_repeats=args.get("num_repeats"),
    )
    return (
        train_images_dir,
    )


def MZ_KohyaSSUseConfig_call(args={}):
    # raise Exception(f"MZ_KohyaSSUseConfig_call: {args}")
    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)

    if workspace_name is None or workspace_name == "":
        raise Exception("工作区名称不能为空(workspace_name is required)")

    workspace_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces", workspace_name)

    if not os.path.exists(workspace_dir):
        raise Exception(f"工作区不存在: {workspace_dir}")

    workspace_config_file = os.path.join(workspace_dir, "config.json")
    train_config_template = args.get("train_config_template", None)
    if not os.path.exists(workspace_config_file):
        train_config_template_dir = args.get("train_config_template_dir", None)

        train_config_template_file = os.path.join(
            train_config_template_dir, train_config_template + ".json")

        shutil.copy(train_config_template_file, workspace_config_file)

    config = None
    with open(workspace_config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        config["metadata"]["train_type"] = train_config_template
        ckpt_name = args.get("ckpt_name", "")
        if ckpt_name == "":
            raise Exception("未选择模型文件(ckpt_name is required)")

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        config["train_config"]["pretrained_model_name_or_path"] = ckpt_path

        # output_dir
        output_dir = os.path.join(workspace_dir, "output")
        config["train_config"]["output_dir"] = output_dir

        datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # output_name
        config["train_config"]["output_name"] = f"{workspace_name}_{train_config_template}_{datetime}"

        dataset_config_path = os.path.join(
            workspace_dir, "dataset.toml")
        config["train_config"]["dataset_config"] = dataset_config_path

        config["train_config"]["max_train_steps"] = str(
            args.get("max_train_steps"))

        config["train_config"]["max_train_epochs"] = str(
            args.get("max_train_epochs"))

        config["train_config"]["save_every_n_epochs"] = str(
            args.get("save_every_n_epochs"))

        config["train_config"]["learning_rate"] = str(
            args.get("learning_rate"))

        advanced_config = args.get("save_advanced_config", {}).copy()

        for k in advanced_config:
            if type(advanced_config[k]) == str and advanced_config[k] == "":
                if k in config["train_config"]:
                    del config["train_config"][k]
                continue
            elif advanced_config[k] == "enable":
                advanced_config[k] = True
            elif advanced_config[k] == "disable":
                advanced_config[k] = False
            else:
                advanced_config[k] = str(advanced_config[k])
            config["train_config"][k] = advanced_config[k]

        # raise Exception(f"args: {json.dumps(config, indent=4)}")

    if config is None:
        raise Exception(f"读取配置文件失败: {workspace_config_file}")

    with open(workspace_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # raise Exception(f"MZ_KohyaSSUseConfig_call: {args}")
    return (
        args,
    )


def config2args(train_parser: argparse.ArgumentParser, config):
    try:
        config_args_list = []
        for key, value in config.items():
            if value is None:
                continue
            if type(value) == bool:
                if value:
                    config_args_list.append(f"--{key}")
            else:
                config_args_list.append(f"--{key}")
                config_args_list.append(str(value))
        args = train_parser.parse_args(config_args_list)
        return args
    except Exception as e:
        raise Exception(f"config2args: {e}")


def check_install():
    try:
        import toml
    except ImportError:
        os.system(f"{sys.executable} -m pip install toml")

    # imagesize
    try:
        import imagesize
    except ImportError:
        os.system(f"{sys.executable} -m pip install imagesize")

    # voluptuous
    try:
        import voluptuous
    except ImportError:
        os.system(f"{sys.executable} -m pip install voluptuous")


import logging


def generate_toml_config(output_path, enable_bucket=True, resolution=512, batch_size=1, image_dir=None, caption_extension=".caption", num_repeats=10, ):
    import toml
    config = {
        'general': {
            'enable_bucket': enable_bucket,
        },
        'datasets': [
            {
                'resolution': resolution,
                'batch_size': batch_size,
                'subsets': [
                    {
                        'image_dir': image_dir,
                        'caption_extension': caption_extension,
                        'num_repeats': num_repeats,
                    },
                ],
            },
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        toml.dump(config, f)


def latent2image(previews, latent_file):
    samples = nodes.LoadLatent().load(latent_file)[0]
    with torch.no_grad():

        samples = samples["samples"].to("cuda")
        preview_bytes = previews.decode_latent_to_preview_image(
            "", samples)
    return preview_bytes[1]


def run_hook_kohya_ss_run_file(kohya_ss_tool_dir, train_config, trainer_func):

    exec_pyfile = os.path.join(os.path.dirname(
        __file__), "hook_kohya_ss_run.py",)
    train_config_str = json.dumps(train_config)
    max_train_steps = train_config.get("max_train_steps")
    is_running = True

    taesd_type = "sd1_5"
    if trainer_func.find("sd1_5") != -1:
        taesd_type = "sd1_5"
    if trainer_func.find("sdxl") != -1:
        taesd_type = "sdxl"
    pb = Utils.progress_bar(train_config.get("max_train_steps"), taesd_type)

    previewer = pb.get_previewer()
    import traceback

    def log_callback(log):
        try:
            resp = json.loads(log)
            if resp.get("type") == "sample_images":
                global_step = resp.get("global_step")
                latent_path = resp.get("latent")
                preview_bytes = None
                if latent_path is not None:
                    preview_bytes = latent2image(previewer, latent_path)
                pb.update(
                    int(global_step), int(max_train_steps), preview_bytes)
            else:
                print(f"LOG: {log}")
        except Exception as e:
            print(f"LOG: {log} e: {e} ")
            print(f"stack: {traceback.format_exc()}")
        return is_running
    server, port = Utils.UDP_Server(log_callback)
    try:
        subprocess.run(
            [sys.executable, exec_pyfile, "--sys_path", kohya_ss_tool_dir,
                "--train_config_json", train_config_str, "--train_func", trainer_func, "--udp_port", str(port)],
            check=True,
        )
        is_running = False
        server.close()
    except Exception as e:
        is_running = False
        server.close()
        raise Exception(f"训练失败!!! 具体报错信息请查看控制台 {e}")


def MZ_KohyaSSTrain_call(args={}):

    # pb = Utils.progress_bar(1)
    # previewer = pb.get_previewer()

    # img = latent2image(previewer, "/tmp/sample_images_4.latent")

    # pb.update(1, 1, img)
    # raise Exception(f"MZ_KohyaSSTrain_call: {args}")
    train_config = args.get("train_config", {})
    workspace_config = train_config.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)
    workspace_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces", workspace_name)

    workspace_config_file = os.path.join(workspace_dir, "config.json")

    if not os.path.exists(workspace_config_file):
        raise Exception(f"配置文件不存在: {workspace_config_file}")

    config = None
    with open(workspace_config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config is None:
        raise Exception(f"读取配置文件失败: {workspace_config_file}")

    kohya_ss_tool_dir = os.path.join(
        Utils.get_minus_zone_models_path(), "train_tools", "kohya_ss_lora")

    if kohya_ss_tool_dir not in sys.path:
        sys.path.append(kohya_ss_tool_dir)
    check_install()

    use_lora = args.get("use_lora", "empty")
    if use_lora == "empty":
        pass
    elif use_lora == "latest":
        workspace_lora_dir = os.path.join(workspace_dir, "output")
        workspace_lora_files = os.listdir(workspace_lora_dir)
        workspace_lora_files = list(
            filter(lambda x: x.endswith(".safetensors"), workspace_lora_files))
        # 排序
        workspace_lora_files = sorted(
            workspace_lora_files, key=lambda x: os.path.getctime(x), reverse=True)
        if len(workspace_lora_files) > 0:
            use_lora = os.path.join(
                workspace_lora_dir, workspace_lora_files[0])
    else:
        pass

    train_config = config.get("train_config")
    if use_lora != "empty" and os.path.exists(use_lora):
        train_config["network_weights"] = use_lora
        train_config["dim_from_weights"] = True

        if "network_dim" in train_config:
            del train_config["network_dim"]
        if "network_alpha" in train_config:
            del train_config["network_alpha"]
        if "network_dropout" in train_config:
            del train_config["network_dropout"]

    train_type = config.get("metadata").get("train_type")
    # 在ComfyUI中无法运行, 梯度有问题, 原因不清楚
    # import train_network
    # train_network.logger = logging.getLogger()
    # trainer = train_network.NetworkTrainer()
    # train_args = config2args(train_network.setup_parser(), train_config)
    # trainer.train(train_args)
    # return

    if train_type == "lora_sd1_5":
        run_hook_kohya_ss_run_file(
            kohya_ss_tool_dir, train_config, "run_lora_sd1_5")
    elif train_type == "lora_sdxl":
        run_hook_kohya_ss_run_file(
            kohya_ss_tool_dir, train_config, "run_lora_sdxl")
    else:
        raise Exception(
            f"暂时不支持的训练类型: {train_type}")

    return (
        "训练完成",
    )
