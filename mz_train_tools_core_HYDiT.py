

import argparse
import csv
import hashlib
import os
import shutil
import sys
import json
import subprocess
import time

import safetensors.torch
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


def MZ_HYDiTInitClone_call(args={}):
    mz_dir = Utils.get_minus_zone_models_path()
    git_url = "https://github.com/Tencent/HunyuanDiT"
    source = args.get("source", "github")
    hunyuan_lora_dir = os.path.join(mz_dir, "train_tools", "HunyuanDiT")
    if git_accelerate_urls.get(source, None) is not None:
        git_url = f"https://{git_accelerate_urls[source]}/Tencent/HunyuanDiT"
    try:
        if not os.path.exists(hunyuan_lora_dir) or not os.path.exists(os.path.join(hunyuan_lora_dir, ".git")):
            subprocess.run(
                ["git", "clone", "--depth", "1", git_url, hunyuan_lora_dir], check=True)

        # 切换远程分支 git remote set-branches origin 'main'
        branch = args.get("branch", "main")

        # 查看本地分支是否一致
        short_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=hunyuan_lora_dir, stdout=subprocess.PIPE, check=True)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=hunyuan_lora_dir, stdout=subprocess.PIPE, check=True)

        short_current_branch = short_result.stdout.decode().strip()
        long_current_branch = result.stdout.decode().strip()
        print(
            f"当前分支(current branch): {long_current_branch}({short_current_branch})")
        print(f"目标分支(target branch): {branch}")
        time.sleep(1)
        if branch != long_current_branch and branch != short_current_branch:
            subprocess.run(
                ["git", "remote", "set-branches", "origin", branch], cwd=hunyuan_lora_dir, check=True)
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", branch], cwd=hunyuan_lora_dir, check=True)

            # 恢复所有文件
            subprocess.run(
                ["git", "checkout", "."], cwd=hunyuan_lora_dir, check=True)

            subprocess.run(
                ["git", "checkout", branch], cwd=hunyuan_lora_dir, check=True)

        content = None
        with open(os.path.join(hunyuan_lora_dir, "hydit/diffusion/pipeline.py"), "r", encoding="utf-8") as f:
            pre_replace = "device = self._execution_device"
            content = f.read()
            content = content.replace(
                pre_replace, "device = torch.device('cuda')")
        with open(os.path.join(hunyuan_lora_dir, "hydit/diffusion/pipeline.py"), "w", encoding="utf-8") as f:
            f.write(content)

    except Exception as e:
        raise Exception(f"克隆kohya-ss/sd-scripts或者切换分支时出现异常,详细信息请查看控制台...")


def MZ_HYDiTInitWorkspace_call(args={}):
    MZ_HYDiTInitClone_call(args)

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

    force_clear = args.get("force_clear") == "enable"
    force_clear_only_images = args.get("force_clear_only_images") == "enable"
    if force_clear:
        if force_clear_only_images:
            images_files = Utils.listdir(train_images_dir)
            for file in images_files:
                if file.lower().endswith(".png") or file.lower().endswith(".jpg"):
                    os.remove(os.path.join(train_images_dir, file))
        else:
            shutil.rmtree(train_images_dir)
            os.makedirs(train_images_dir, exist_ok=True)

    saved_images_path = []

    for i, pil_image in enumerate(pil_images):
        pil_image = Utils.resize_max(pil_image, resolution, resolution)
        width, height = pil_image.size
        filename = hashlib.md5(pil_image.tobytes()).hexdigest() + ".png"
        pil_image.save(os.path.join(train_images_dir, filename))
        saved_images_path.append(filename)

    same_caption_generate = args.get("same_caption_generate") == "enable"
    if same_caption_generate:
        same_caption = args.get("same_caption").strip()
        if same_caption != "":
            # 循环已经保存的图片
            for i, filename in enumerate(saved_images_path):
                base_filename = os.path.splitext(filename)[0]
                caption_filename = base_filename + ".caption"
                with open(os.path.join(train_images_dir, caption_filename), "w", encoding="utf-8") as f:
                    f.write(same_caption)

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
        if not os.path.exists(download_fullpath):
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
            if not os.path.exists(download_fullpath):
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
            if not os.path.exists(download_fullpath):
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
            if not os.path.exists(download_fullpath):
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
            if not os.path.exists(download_fullpath):
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

    advanced_config = args.get("advanced_config", {})
    fix_float_keys = ["lr", "warmup_min_lr",
                      "warmup_max_lr", "weight_decay", "warmup_num_steps",]
    for key in fix_float_keys:
        if key in advanced_config:
            if isinstance(advanced_config[key], str):
                advanced_config[key] = float(advanced_config[key])

    args = check_model_auto_download(args)
    # raise Exception(args)
    resolution = args.get("resolution")

    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)
    workspace_dir = workspace_config.get("workspace_dir", None)

    workspace_images_dir = args.get("workspace_images_dir", None)
    if workspace_images_dir is None:
        workspace_images_dir = os.path.join(workspace_dir, "train_images")

    full_filenames = Utils.listdir(workspace_images_dir)

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
            print(f"处理文件: {filename}")
            if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):

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
    if os.path.exists(arrows_dir):
        shutil.rmtree(arrows_dir)

    os.makedirs(arrows_dir, exist_ok=True)
    csv2arrow_exec = os.path.join(
        HYDiT_tool_dir, "hydit", "data_loader", "csv2arrow.py")
    try:
        subprocess.run([
            sys.executable, csv2arrow_exec, csv_filename, arrows_dir
        ], check=True)
    except Exception as e:
        raise Exception(f"生成arrow文件时出现异常,详细信息请查看控制台...")

    arrows_files = Utils.listdir(arrows_dir)

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

    lora_ckpt = args.get("base_lora")
    if lora_ckpt == "empty":
        lora_ckpt = None
    elif lora_ckpt == "latest":
        lora_ckpt = None
        loras = search_loras([workspace_dir])
        if len(loras) > 0:
            lora_ckpt = loras[0]
    elif not os.path.exists(lora_ckpt):
        raise Exception(f"未找到指定的lora文件: {lora_ckpt}")

    print(f"使用指定的lora文件: {lora_ckpt}")
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
        "t5_encoder_path": args.get("t5_encoder_path") if args.get("t5_encoder_path") != "none" else None,
        "results_dir": output_dir,
        "task_flag": output_name,
        "lora_ckpt": lora_ckpt,
        "rank": int(args.get("rank")),
    })

    if "target_modules" in train_config:
        del train_config["target_modules"]
    for key in advanced_config:
        if key.startswith("target_modules_"):
            if advanced_config[key] == "enable":
                if "target_modules" not in train_config:
                    train_config["target_modules"] = []
                train_config["target_modules"].append(
                    key.replace("target_modules_", ""))
        elif advanced_config[key] == "enable" or advanced_config[key] == "disable":
            train_config[key] = advanced_config[key] == "enable"
        else:
            train_config[key] = advanced_config[key]

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
        image_files = Utils.listdir(sample_images_dir)

        image_files = list(
            filter(lambda x: x.lower().endswith(".png"), image_files))

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


def get_HunYuanDiT_model_from_path(model_path, lora_path, width, height):

    from hydit.modules.models import HUNYUAN_DIT_CONFIG
    import importlib
    import hydit.modules.models
    importlib.reload(hydit.modules.models)
    HunYuanDiT = hydit.modules.models.HunYuanDiT

    from types import SimpleNamespace

    # self.text_states_dim = args.text_states_dim
    # self.text_states_dim_t5 = args.text_states_dim_t5
    # self.text_len = args.text_len
    # self.text_len_t5 = args.text_len_t5
    # self.norm = args.norm
    # use_flash_attn = args.infer_mode == 'fa' or args.use_flash_attn

    args = SimpleNamespace()
    args.learn_sigma = True
    args.text_states_dim = 1024
    args.text_states_dim_t5 = 2048
    args.text_len = 77
    args.text_len_t5 = 256
    args.norm = "layer"
    args.infer_mode = "torch"
    args.use_fp16 = True
    args.qk_norm = True
    args.use_flash_attn = False

    model_config = HUNYUAN_DIT_CONFIG["DiT-g/2"]

    latent_size = (width // 8, height // 8)

    model = HunYuanDiT(args,
                       input_size=latent_size,
                       **model_config,
                       ).half().to("cuda")
    state_dict = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=False)

    if lora_path is not None:
        from peft import PeftConfig

        # 查询后缀
        if lora_path.lower().endswith(".safetensors"):
            from safetensors.torch import load_file
            adapter_state_dict = load_file(lora_path)
        else:
            adapter_state_dict = torch.load(lora_path)

        peft_config = PeftConfig()

        # 相同目录下同名的json文件
        lora_json_path = os.path.splitext(lora_path)[0] + ".json"
        if not os.path.exists(lora_json_path):
            raise Exception(f"未找到对应的json文件: {lora_json_path}")
        with open(lora_json_path, "r", encoding="utf-8") as f:
            loaded_attributes = json.load(f)

        for key, value in loaded_attributes.items():
            setattr(peft_config, key, value)

        model.load_adapter(
            adapter_state_dict=adapter_state_dict,
            peft_config=peft_config,
        )
        model.merge_and_unload()

    return model


def clean_unet():
    unet_data = Utils.cache_get("HYDiT_UNET")
    if unet_data is not None:
        unet_data["model"].cpu()
        Utils.cache_set("HYDiT_UNET", None)
        torch.cuda.empty_cache()


def clean_vae():
    vae_data = Utils.cache_get("HYDiT_VAE_EMA")
    if vae_data is not None:
        vae_data["model"].cpu()
        Utils.cache_set("HYDiT_VAE_EM", None)
        torch.cuda.empty_cache()


def clean_clip_text_encoder():
    clip_text_encoder = Utils.cache_get("HYDiT_CLIP_TEXT_ENCODER")
    if clip_text_encoder is not None:
        clip_text_encoder["model"].cpu()
        Utils.cache_set("HYDiT_CLIP_TEXT_ENCODER", None)
        torch.cuda.empty_cache()


def clean_t5_encoder():
    t5_encoder = Utils.cache_get("HYDiT_T5_ENCODER")
    if t5_encoder is not None:
        t5_encoder["model"].cpu()
        Utils.cache_set("HYDiT_T5_ENCODER", None)
        torch.cuda.empty_cache()


from torch import nn


class CustomizeEmbedsModel(nn.Module):
    dtype = torch.float16
    x = torch.zeros(1, 1, 256, 2048)

    def __init__(self, *args, **kwargs):
        super().__init__()

    def to(self, *args, **kwargs):
        self.dtype = torch.float16
        return self

    def forward(self, *args, **kwargs):
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

    def __init__(self):
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
        self.model = CustomizeEmbedsModel()
        self.max_length = 256


def MZ_HYDiTSimpleT2I_call(args={}):
    check_required()
    MZ_HYDiTInitClone_call(args)
    args = check_model_auto_download(args)

    HYDiT_tool_dir = os.path.join(
        Utils.get_minus_zone_models_path(), "train_tools", "HunyuanDiT")
    if HYDiT_tool_dir not in sys.path:
        print(f"add {HYDiT_tool_dir} to sys.path")
        sys.path.append(HYDiT_tool_dir)

    from hydit.inference import SAMPLER_FACTORY, HUNYUAN_DIT_CONFIG, get_fill_resize_and_crop, get_2d_rotary_pos_embed
    from hydit.modules.text_encoder import MT5Embedder
    from hydit.utils.tools import set_seeds
    from PIL import Image
    from diffusers import schedulers, AutoencoderKL
    from transformers import BertModel, BertTokenizer

    SAMPLER_FACTORY["uni_pc"] = {
        'scheduler': 'UniPCMultistepScheduler',
        'name': 'UniPCMultistepScheduler',
        'kwargs': {
            'beta_schedule': 'scaled_linear',
            'beta_start': 0.00085,
            'beta_end': 0.03,
            'prediction_type': 'v_prediction',
            'trained_betas': None,
            'solver_order': 2,
            # 'algorithm_type': 'dpmsolver++',
        }
    }
    SAMPLER_FACTORY["dpmpp_2m_karras"] = {
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

    width = args.get("width")
    height = args.get("height")

    unet_path = args.get("unet_path")
    lora_path = args.get("lora_path")
    unet_cache_key = f"HYDiT_UNET"
    unet_data = Utils.cache_get(unet_cache_key)
    if unet_data is None or unet_data.get("unet_path") != unet_path or unet_data.get("lora_path") != lora_path:
        clean_unet()
        _lora_path = lora_path if lora_path != "none" else None
        unet = get_HunYuanDiT_model_from_path(
            args.get("unet_path"), _lora_path, width, height)
        Utils.cache_set(unet_cache_key, {
            "model": unet,
            "unet_path": unet_path,
            "lora_path": lora_path,
        })
    else:
        unet = unet_data["model"]

    text_encoder_path = args.get("text_encoder_path")
    clip_text_encoder_data = Utils.cache_get("HYDiT_CLIP_TEXT_ENCODER")

    if clip_text_encoder_data is None or clip_text_encoder_data.get("text_encoder_path") != text_encoder_path:
        clean_clip_text_encoder()
        clip_text_encoder = BertModel.from_pretrained(
            text_encoder_path, False, revision=None, local_files_only=True).to("cuda")
        Utils.cache_set("HYDiT_CLIP_TEXT_ENCODER", {
            "model": clip_text_encoder,
            "text_encoder_path": text_encoder_path,
        })
    else:
        clip_text_encoder = clip_text_encoder_data["model"]

    tokenizer_path = args.get("tokenizer_path")
    text_tokenizer = BertTokenizer.from_pretrained(
        tokenizer_path, revision=None, local_files_only=True)

    vae_ema_path = args.get("vae_ema_path")
    vae_ema_data = Utils.cache_get("HYDiT_VAE_EMA")
    if vae_ema_data is None or vae_ema_data.get("vae_ema_path") != vae_ema_path:
        clean_vae()
        if os.path.isdir(vae_ema_path):
            vae = AutoencoderKL.from_pretrained(
                vae_ema_path, local_files_only=True).to("cuda")
        else:
            vae = AutoencoderKL.from_single_file(
                vae_ema_path, local_files_only=True).to("cuda")

        Utils.cache_set("HYDiT_VAE_EMA", {
            "model": vae,
            "vae_ema_path": vae_ema_path,
        })
    else:
        vae = vae_ema_data["model"]

    t5_encoder_path = args.get("t5_encoder_path")
    if t5_encoder_path == "none":
        t5_encoder = CustomizeEmbeds()
    else:
        t5_encoder_data = Utils.cache_get("HYDiT_T5_ENCODER")
        if t5_encoder_data is None or t5_encoder_data.get("t5_encoder_path") != t5_encoder_path:
            clean_t5_encoder()
            t5_encoder = MT5Embedder(
                t5_encoder_path, torch_dtype=torch.float16, max_length=256)
            Utils.cache_set("HYDiT_T5_ENCODER", {
                "model": t5_encoder,
                "t5_encoder_path": t5_encoder_path,
            })
        else:
            t5_encoder = t5_encoder_data["model"]

    sampler = args.get("scheduler")
    scheduler = SAMPLER_FACTORY[sampler]['scheduler']
    kwargs = SAMPLER_FACTORY[sampler]['kwargs']
    # Build scheduler according to the sampler.
    scheduler_class = getattr(schedulers, scheduler)
    scheduler = scheduler_class(**kwargs)

    scheduler.set_timesteps(args.get("steps"), "cuda")

    from hydit.diffusion.pipeline import StableDiffusionPipeline
    pipeline = StableDiffusionPipeline(vae=vae,
                                       text_encoder=clip_text_encoder,
                                       tokenizer=text_tokenizer,
                                       unet=unet,
                                       scheduler=scheduler,
                                       feature_extractor=None,
                                       safety_checker=None,
                                       requires_safety_checker=False,
                                       progress_bar_config=None,
                                       embedder_t5=t5_encoder,
                                       infer_mode="torch",
                                       )

    pipeline = pipeline.to("cuda")

    seed = args.get("seed", 0)
    target_height = int((height // 16) * 16)
    target_width = int((width // 16) * 16)
    prompt = args.get("prompt", "").strip()
    negative_prompt = args.get("negative_prompt", "").strip()

    batch_size = 1
    style = torch.as_tensor([0, 0] * batch_size, device="cuda")

    src_size_cond = (target_width, target_height)
    size_cond = list(src_size_cond) + [target_width, target_height, 0, 0]
    image_meta_size = torch.as_tensor(
        [size_cond] * 2 * batch_size, device="cuda",)

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

    freqs_cis_img = calc_rope(target_height, target_width)
    guidance_scale = args.get("cfg", 7.0)
    infer_steps = args.get("steps", 20)

    with torch.inference_mode():
        with torch.cuda.amp.autocast():

            pbar = Utils.progress_bar(0, "sdxl")
            preview = pbar.get_previewer()

            def callback(step: int, timestep: int, latents: torch.FloatTensor, pred_x0):
                try:
                    pil_img = preview.decode_latent_to_preview_image(
                        None,
                        latents,
                    )[1]
                    pbar.update(step, infer_steps, pil_img)
                except Exception as e:
                    print("decode_tensors error:", e)
            generator = set_seeds(seed, device="cuda")
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
                generator=generator,
                use_fp16=True,
                learn_sigma=True,
                freqs_cis_img=freqs_cis_img,
                image_meta_size=image_meta_size,
                callback=callback,
                callback_steps=2,
            )[0]

            if type(samples) == list:
                pil_image = samples[0]
            else:
                pil_image = samples

            tensor_image = Utils.pil2tensor(pil_image)

            keep_device = args.get("keep_device", "disable") == "enable"
            if not keep_device:
                del pipeline
                del unet
                del clip_text_encoder
                del vae
                del t5_encoder
                clean_unet()
                clean_clip_text_encoder()
                clean_vae()
                torch.cuda.empty_cache()

            return (Utils.list_tensor2tensor([tensor_image]),)


def search_loras(wdirs):
    loras = []
    for wdir in wdirs:
        for root, dirs, files in os.walk(wdir):
            # 排除隐藏文件夹
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            # 存在adapter_config.json文件并且存在adapter_model.safetensors文件
            for file in files:
                if file == "adapter_config.json":
                    lora_path = os.path.join(root, "adapter_model.safetensors")
                    if os.path.exists(lora_path):
                        loras.append(root)

    # 按时间倒序
    loras = sorted(loras, key=lambda x: os.path.getmtime(x), reverse=True)
    return loras
