

import argparse
import csv
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


git_accelerate_urls = {
    "githubfast": "githubfast.com",
    "521github": "521github.com",
    "kkgithub": "kkgithub.com",
}

# 初始化工具仓库和工作区


def MZ_HYDiTInitWorkspace_call(args={}):
    mz_dir = Utils.get_minus_zone_models_path()
    git_url = "https://github.com/Tencent/HunyuanDiT"
    source = args.get("source", "github")
    kohya_ss_lora_dir = os.path.join(mz_dir, "train_tools", "HunyuanDiT")
    if git_accelerate_urls.get(source, None) is not None:
        git_url = f"https://{git_accelerate_urls[source]}/Tencent/HunyuanDiT"
    try:
        if not os.path.exists(kohya_ss_lora_dir) or not os.path.exists(os.path.join(kohya_ss_lora_dir, ".git")):
            subprocess.run(
                ["git", "clone", "--depth", "1", git_url, kohya_ss_lora_dir], check=True)

        # 切换远程分支 git remote set-branches origin 'main'
        branch = args.get("branch", "main")

        # 查看本地分支是否一致
        short_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=kohya_ss_lora_dir, stdout=subprocess.PIPE, check=True)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=kohya_ss_lora_dir, stdout=subprocess.PIPE, check=True)

        short_current_branch = short_result.stdout.decode().strip()
        long_current_branch = result.stdout.decode().strip()
        print(
            f"当前分支(current branch): {long_current_branch}({short_current_branch})")
        print(f"目标分支(target branch): {branch}")

        if branch != result.stdout.decode() and branch != short_result.stdout.decode():
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

    workspace_name = args.get("train_name", None)
    workspace_name = workspace_name.strip()

    if workspace_name is None or workspace_name == "":
        raise Exception("训练名称不能为空(train_name is required)")

    args["workspace_name"] = workspace_name
    workspaces_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces")

    os.makedirs(workspaces_dir, exist_ok=True)

    workspace_dir = os.path.join(workspaces_dir, workspace_name)
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)

    args["workspace_dir"] = workspace_dir

    return (
        args,
    )


def MZ_HYDiTDatasetConfig_call(args={}):
    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)
    workspace_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces", workspace_name)
    train_images_dir = os.path.join(workspace_dir, "train_images")
    os.makedirs(train_images_dir, exist_ok=True)

    images = args.get("images", None)

    if images is None:
        print(f"训练数据位于:{train_images_dir},未检测到传入图片默认已有训练数据,将直接跳过数据准备步骤...")
        return (
            train_images_dir,
        )

    pil_images = Utils.tensors2pil_list(images)

    resolution = args.get("resolution", 1024)

    if workspace_name is None or workspace_name == "":
        raise Exception("训练名称不能为空(workspace_name is required)")

    saved_images_path = []

    for i, pil_image in enumerate(pil_images):
        pil_image = Utils.resize_max(pil_image, resolution, resolution)
        width, height = pil_image.size
        filename = hashlib.md5(pil_image.tobytes()).hexdigest() + ".png"
        pil_image.save(os.path.join(train_images_dir, filename))
        saved_images_path.append(filename)

    return (
        train_images_dir,
    )


HYDiT_MODEL = {
    "HunyuanDiT/t2i/model/pytorch_model_ema.pt": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmodel%2Fpytorch_model_ema.pt",
    },
    "HunyuanDiT/t2i/model/pytorch_model_module.pt": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmodel%2Fpytorch_model_module.pt",
    },
    "HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fsdxl-vae-fp16-fix%2Fdiffusion_pytorch_model.safetensors",
    },
    "HunyuanDiT/t2i/sdxl-vae-fp16-fix/config.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fsdxl-vae-fp16-fix%2Fconfig.json",
    },
    "HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fclip_text_encoder%2Fpytorch_model.bin",
    },
    "HunyuanDiT/t2i/clip_text_encoder/config.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fclip_text_encoder%2Fconfig.json",
    },
    "HunyuanDiT/t2i/tokenizer/special_tokens_map.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Ftokenizer%2Fspecial_tokens_map.json",
    },
    "HunyuanDiT/t2i/tokenizer/tokenizer_config.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Ftokenizer%2Ftokenizer_config.json",
    },
    "HunyuanDiT/t2i/tokenizer/vocab.txt": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Ftokenizer%2Fvocab.txt",
    },
    "HunyuanDiT/t2i/tokenizer/vocab_org.txt": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Ftokenizer%2Fvocab_org.txt",
    },
    "HunyuanDiT/t2i/mt5/config.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmt5%2Fconfig.json",
    },
    "HunyuanDiT/t2i/mt5/generation_config.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmt5%2Fgeneration_config.json",
    },
    "HunyuanDiT/t2i/mt5/special_tokens_map.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmt5%2Fspecial_tokens_map.json",
    },
    "HunyuanDiT/t2i/mt5/spiece.model": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmt5%2Fspiece.model",
    },
    "HunyuanDiT/t2i/mt5/tokenizer_config.json": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmt5%2Ftokenizer_config.json",
    },
    "HunyuanDiT/t2i/mt5/pytorch_model.bin": {
        "url": "https://www.modelscope.cn/api/v1/models/modelscope/HunyuanDiT/repo?Revision=master&FilePath=t2i%2Fmt5%2Fpytorch_model.bin",
    },
}


def check_model_auto_download(args):
    unet_path = args.get("unet_path", "auto")
    hunyuan_base_path = os.path.join(
        Utils.get_comfyui_models_path(), "hunyuan")
    if unet_path == "auto":
        ema_to_module = args.get("ema_to_module", "enable") == "enable"
        if ema_to_module:
            download_file = "HunyuanDiT/t2i/model/pytorch_model_ema.pt"
        else:
            download_file = "HunyuanDiT/t2i/model/pytorch_model_module.pt"
        download_fullpath = os.path.join(
            hunyuan_base_path, download_file)
        os.makedirs(os.path.dirname(download_fullpath), exist_ok=True)

        if os.path.exists(download_fullpath):
            args["unet_path"] = download_fullpath
        else:
            success_path = Utils.download_file(
                HYDiT_MODEL[download_file]["url"],
                download_fullpath,
            )
            if os.path.exists(success_path):
                args["unet_path"] = download_fullpath

    vae_ema_path = args.get("vae_ema_path", "auto")
    if vae_ema_path == "auto":
        download_files = []
        prefix = "HunyuanDiT/t2i/sdxl-vae-fp16-fix/"
        for key in HYDiT_MODEL:
            if key.startswith(prefix):
                download_files.append(key)

        for download_file in download_files:
            download_fullpath = os.path.join(
                hunyuan_base_path, download_file)
            os.makedirs(os.path.dirname(download_fullpath), exist_ok=True)

            if not os.path.exists(download_fullpath):
                Utils.download_file(
                    HYDiT_MODEL[download_file]["url"],
                    download_fullpath,
                )

        args["vae_ema_path"] = os.path.join(
            hunyuan_base_path, prefix)

    text_encoder_path = args.get("text_encoder_path", "auto")
    if text_encoder_path == "auto":
        download_files = []
        prefix = "HunyuanDiT/t2i/clip_text_encoder/"
        for key in HYDiT_MODEL:
            if key.startswith(prefix):
                download_files.append(key)

        for download_file in download_files:
            download_fullpath = os.path.join(
                hunyuan_base_path, download_file)
            os.makedirs(os.path.dirname(download_fullpath), exist_ok=True)

            if not os.path.exists(download_fullpath):
                Utils.download_file(
                    HYDiT_MODEL[download_file]["url"],
                    download_fullpath,
                )

        args["text_encoder_path"] = os.path.join(
            hunyuan_base_path, prefix)

    tokenizer_path = args.get("tokenizer_path", "auto")
    if tokenizer_path == "auto":
        download_files = []
        prefix = "HunyuanDiT/t2i/tokenizer/"
        for key in HYDiT_MODEL:
            if key.startswith(prefix):
                download_files.append(key)

        for download_file in download_files:
            download_fullpath = os.path.join(
                hunyuan_base_path, download_file)
            os.makedirs(os.path.dirname(download_fullpath), exist_ok=True)

            if not os.path.exists(download_fullpath):
                Utils.download_file(
                    HYDiT_MODEL[download_file]["url"],
                    download_fullpath,
                )

        args["tokenizer_path"] = os.path.join(
            hunyuan_base_path, prefix)

    t5_encoder_path = args.get("t5_encoder_path", "auto")
    if t5_encoder_path == "auto":
        download_files = []
        prefix = "HunyuanDiT/t2i/mt5/"
        for key in HYDiT_MODEL:
            if key.startswith(prefix):
                download_files.append(key)

        for download_file in download_files:
            download_fullpath = os.path.join(
                hunyuan_base_path, download_file)
            os.makedirs(os.path.dirname(download_fullpath), exist_ok=True)

            if not os.path.exists(download_fullpath):
                Utils.download_file(
                    HYDiT_MODEL[download_file]["url"],
                    download_fullpath,
                )

        args["t5_encoder_path"] = os.path.join(
            hunyuan_base_path, prefix)

    return args


def check_required():
    packages = [
        "pandas", "pyarrow", "diffusers", "transformers",
        "timm", "peft", "accelerate", "loguru", "einops", "sentencepiece",
        "polygraphy", "protobuf"
    ]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip",
                           "install", package], check=True)

    try:
        import deepspeed
    except ImportError:
        raise Exception("请手动安装合适的deepspeed版本")


def MZ_HYDiTTrain_call(args={}):
    check_required()
    args = check_model_auto_download(args)
    # raise Exception(args)
    resolution = args.get("resolution")

    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)
    workspace_dir = workspace_config.get("workspace_dir", None)

    workspace_images_dir = args.get("workspace_images_dir", None)
    if workspace_images_dir is None:
        workspace_images_dir = os.path.join(workspace_dir, "train_images")

    full_filenames = os.listdir(workspace_images_dir)

    # 创建image_text.csv文件
    csv_filename = os.path.join(workspace_dir, "image_text.csv")

    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    max_width, max_height, min_width, min_height = 0, 0, 99999, 99999

    from PIL import Image

    with open(csv_filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["image_path", "text_zh"])
        for filename in full_filenames:
            if filename.endswith(".png"):

                image_path = os.path.join(workspace_images_dir, filename)
                pil_image = Image.open(image_path)
                width, height = pil_image.size
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height
                if width < min_width:
                    min_width = width
                if height < min_height:
                    min_height = height

                text_zh = ""

                # 去除后缀
                _caption_filename = filename.split(".")[0]
                caption_filename = ""
                if os.path.exists(os.path.join(workspace_images_dir, _caption_filename + ".txt")):
                    caption_filename = _caption_filename + ".txt"
                elif os.path.exists(os.path.join(workspace_images_dir, _caption_filename + ".caption")):
                    caption_filename = _caption_filename + ".caption"
                else:
                    caption_filename = ""
                if caption_filename != "":
                    with open(os.path.join(workspace_images_dir, caption_filename), "r") as f:
                        caption = f.read()
                        text_zh = caption

                writer.writerow([image_path, text_zh])

    if resolution > max_width and resolution > max_height:
        print("resolution 大于所有图片的宽高,强制使用最大宽高作为resolution")
        resolution = max(max_width, max_height) // 8 * 8

    print(f"csv文件已生成: {csv_filename}")

    HYDiT_tool_dir = os.path.join(
        Utils.get_minus_zone_models_path(), "train_tools", "HunyuanDiT")

    # python ./hydit/data_loader/csv2arrow.py ./dataset/porcelain/csvfile/image_text.csv ./dataset/porcelain/arrows
    arrows_dir = os.path.join(workspace_dir, "arrows")
    os.makedirs(arrows_dir, exist_ok=True)
    csv2arrow_exec = os.path.join(
        HYDiT_tool_dir, "hydit", "data_loader", "csv2arrow.py")
    try:
        subprocess.run([
            sys.executable, csv2arrow_exec, csv_filename, arrows_dir
        ], check=True)
    except Exception as e:
        raise Exception(f"生成arrow文件时出现异常,详细信息请查看控制台...")

    arrows_files = os.listdir(arrows_dir)

    dataset_yaml_path = os.path.join(workspace_dir, "dataset.yaml")

    import yaml
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "source": [os.path.join(arrows_dir, arrow_file) for arrow_file in arrows_files],
        }, f)

    print(f"dataset.yaml文件已生成: {dataset_yaml_path}")

    idk_exec = os.path.join(
        os.path.dirname(__file__), "hook_HYDiT_idk_run.py")

    try:
        from index_kits import __version__
    except Exception as e:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", os.path.join(
                HYDiT_tool_dir, "IndexKits")
        ], check=True)
    # idk base -c dataset/yamls/porcelain.yaml -t dataset/porcelain/jsons/porcelain.json

    dataset_json = os.path.join(workspace_dir, "dataset.json")
    try:
        subprocess.run([
            sys.executable, idk_exec, "base", "-c", dataset_yaml_path, "-t", dataset_json,
        ], check=True)
    except Exception as e:
        raise Exception(f"生成index文件时出现异常,详细信息请查看控制台...")

    print(f"index文件已生成: {dataset_json}")

    dataset_mt_yaml_path = os.path.join(workspace_dir, "dataset_mt.yaml")

    with open(dataset_mt_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "src": [dataset_json],
            "base_size": resolution,
            "target_ratios": ["1:1", "3:4", "4:3", "16:9", "9:16"],
        }, f)

    print(f"dataset_mt.yaml文件已生成: {dataset_mt_yaml_path}")

    #  idk multireso -c dataset/yamls/porcelain_mt.yaml -t dataset/porcelain/jsons/porcelain_mt.json

    dataset_mt_json = os.path.join(workspace_dir, "dataset_mt.json")
    try:
        subprocess.run([
            sys.executable, idk_exec, "multireso", "-c", dataset_mt_yaml_path, "-t", dataset_mt_json,
        ], check=True)
    except Exception as e:
        raise Exception(f"生成多分辨率index文件时出现异常,详细信息请查看控制台...")

    print(f"多分辨率index文件已生成: {dataset_mt_json}")

    train_config = {}

    train_config_path = os.path.join(workspace_dir, "mz_train_config.json")

    if os.path.exists(train_config_path):
        with open(train_config_path, "r", encoding="utf-8") as f:
            train_config = json.load(f)

# "unet": (["auto"] + models + unet_models, {"default": "auto"}),
#                 "vae": (["auto"] + folders + vae_models, {"default": "auto"}),
#                 "text_encoder": (["auto"] + folders, {"default": "auto"}),
#                 "tokenizer": (["auto"] + folders, {"default": "auto"}),
#                 "t5_encoder": (["auto"] + folders, {"default": "auto"}),

    epochs = args.get("epochs", 50)
    output_dir = os.path.join(workspace_dir, "output")

    datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_name = f"{workspace_name}_{datetime}"

    train_config.update({
        "workspace_name": workspace_name,
        "workspace_dir": workspace_dir,
        "batch_size": args.get("batch_size", 1),
        "ema_to_module": args.get("ema_to_module", "enable") == "enable",
        "target_ratios": train_config.get("target_ratios", ["1:1", "3:4", "4:3", "16:9", "9:16"]),
        "index_file": dataset_mt_json,
        "epochs": epochs,
        "ckpt_every": int(args.get("ckpt_every")),
        "rope_img": f"base{resolution}",
        "rope_real": args.get("rope_real", "enable") == "enable",
        "image_size": resolution,
        "unet_path": args.get("unet_path"),
        "vae_ema_path": args.get("vae_ema_path"),
        "text_encoder_path": args.get("text_encoder_path"),
        "tokenizer_path": args.get("tokenizer_path"),
        "t5_encoder_path": args.get("t5_encoder_path"),
        "results_dir": output_dir,
        "task_flag": output_name,
    })

    sample_generate = args.get("sample_generate", "enable")
    sample_prompt = args.get("sample_prompt", "")
    sample_config_file = os.path.join(workspace_dir, "sample_config.json")
    if sample_generate == "enable":
        resolution = args.get("resolution")
        sample_config = {
            "prompt": sample_prompt,
            "negative_prompt": "",
            "cfg": 5.0,
            "steps": 20,
            "width": resolution,
            "height": resolution,
        }
        with open(sample_config_file, "w", encoding="utf-8") as f:
            json.dump(sample_config, f, indent=4, ensure_ascii=False)

        train_config["sample_config_file"] = sample_config_file
    else:
        train_config["sample_config_file"] = None

    with open(train_config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=4, ensure_ascii=False)

    run_hook_HYDiT_pyexec(train_config_path, train_config, HYDiT_tool_dir)
    return (
        args,
    )


def run_hook_HYDiT_pyexec(train_config_path, train_config, HYDiT_tool_dir):
    pb = Utils.progress_bar(0, "sdxl")
    import traceback
    import comfy.model_management
    stop_server = None

    is_running = True

    def log_callback(log):
        try:
            comfy.model_management.throw_exception_if_processing_interrupted()
        except Exception as e:
            stop_server()
            return is_running

        try:
            resp = log
            if resp.get("type") == "sample_images":
                total_steps = resp.get("total_steps")
                global_step = resp.get("global_step")
                xy_img = get_sample_images(train_config)
                if xy_img is None:
                    pb.update(
                        int(global_step), int(total_steps), None)
                    return is_running
                max_side = max(xy_img.width, xy_img.height)

                pb.update(
                    int(global_step), int(total_steps), ("JPEG", xy_img, max_side))
            else:
                print(f"LOG: {log}")
        except Exception as e:
            print(f"LOG: {log} e: {e} ")
            print(f"stack: {traceback.format_exc()}")
        return is_running
    stop_server, port = Utils.Simple_Server(log_callback)

    exec_pyfile = os.path.join(os.path.dirname(
        __file__), "hook_HYDiT_run.py",)

    try:
        subprocess.run(
            [sys.executable, "-m", "torch.distributed.launch", "--use_env", exec_pyfile, "--sys_path", HYDiT_tool_dir,
                "--mz_master_port", str(port), "--train_config_file", train_config_path],
            check=True,
        )
        stop_server()
        is_running = False
    except Exception as e:
        stop_server()
        is_running = False
        raise Exception(f"训练失败!!! 具体报错信息请查看控制台...")


def get_sample_images(train_config):
    if train_config.get("sample_config_file") is None:
        return None

    from PIL import Image
    output_name = train_config.get("task_flag")

    workspace_dir = train_config.get("workspace_dir")

    sample_images_dir = os.path.join(workspace_dir, "sample_images")

    pil_images = []
    pre_render_texts_x = []
    if os.path.exists(sample_images_dir):
        image_files = os.listdir(sample_images_dir)
        image_files = list(
            filter(lambda x: x.endswith(".png"), image_files))
        # 筛选 output_name 前缀
        image_files = list(
            filter(lambda x: x.startswith(output_name), image_files))

        image_files = sorted(image_files, key=lambda x: x)

        for image_file in image_files:
            pil_image = Image.open(os.path.join(sample_images_dir, image_file))
            pil_images.append([pil_image])
            pre_render_texts_x.append(image_file)
    result = Utils.xy_image(
        pre_render_images=pil_images,
        pre_render_texts_x=pre_render_texts_x,
        pre_render_texts_y=[""],
    )
    return result
    return None
