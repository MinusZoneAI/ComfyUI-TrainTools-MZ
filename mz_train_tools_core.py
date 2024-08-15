

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
    Utils.clone_repo(args)

    workspace_name = args.get("lora_name", None)
    workspace_name = workspace_name.strip()

    if workspace_name is None or workspace_name == "":
        raise Exception("lora名称不能为空(lora_name is required)")

    args["workspace_name"] = workspace_name
    workspaces_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces")

    os.makedirs(workspaces_dir, exist_ok=True)

    workspace_dir = os.path.join(workspaces_dir, workspace_name)
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)

    return (
        args,
    )


def MZ_KohyaSSDatasetConfig_call(args={}):
    images = args.get("images")
    pil_images = Utils.tensors2pil_list(images)

    conditioning_images = args.get("conditioning_images", None)
    conditioning_pil_images = None
    if conditioning_images is not None:
        conditioning_pil_images = Utils.tensors2pil_list(
            conditioning_images)

    resolution = args.get("resolution", 512)

    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)

    if workspace_name is None or workspace_name == "":
        raise Exception("lora名称不能为空(lora_name is required)")

    workspace_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces", workspace_name)
    train_images_dir = os.path.join(workspace_dir, "train_images")
    os.makedirs(train_images_dir, exist_ok=True)

    conditioning_images_dir = os.path.join(
        workspace_dir, "conditioning_images")

    force_clear = args.get("force_clear") == "enable"
    force_clear_only_images = args.get("force_clear_only_images") == "enable"
    if force_clear:
        if force_clear_only_images:
            images_files = Utils.listdir(train_images_dir)
            for file in images_files:
                if file.lower().endswith(".png") or file.lower().endswith(".jpg") or file.lower().endswith(".webp"):
                    os.remove(os.path.join(train_images_dir, file))
        else:
            shutil.rmtree(train_images_dir)
            os.makedirs(train_images_dir, exist_ok=True)

    image_format = args.get("image_format")
    file_extension = "." + image_format
    saved_images_path = []
    for i, pil_image in enumerate(pil_images):
        pil_image = Utils.resize_max(pil_image, resolution, resolution)
        width, height = pil_image.size
        filename = hashlib.md5(
            pil_image.tobytes()).hexdigest() + file_extension
        pil_image.save(os.path.join(train_images_dir, filename))
        saved_images_path.append(filename)

        if conditioning_pil_images is not None:
            os.makedirs(conditioning_images_dir, exist_ok=True)
            conditioning_pil_images[i].resize((width, height)).save(
                os.path.join(conditioning_images_dir, filename))

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

    if conditioning_images is None:
        conditioning_images_dir = None

    caption_extension = args.get("caption_extension", ".caption")

    if os.path.exists(os.path.join(workspace_dir, "dataset.json")):
        os.remove(os.path.join(workspace_dir, "dataset.json"))

    if os.path.exists(os.path.join(workspace_dir, "dataset.toml")):
        os.remove(os.path.join(workspace_dir, "dataset.toml"))

    dataset_config_extension = args.get("dataset_config_extension")
    generate_dataset_config(
        workspace_dir=workspace_dir,
        output_path=os.path.join(
            workspace_dir, "dataset" + dataset_config_extension),
        enable_bucket=args.get("enable_bucket") == "enable",
        resolution=args.get("resolution"),
        batch_size=args.get("batch_size"),
        image_dir=train_images_dir,
        conditioning_data_dir=conditioning_images_dir,
        caption_extension=caption_extension,
        num_repeats=args.get("num_repeats"),
        is_reg=args.get("auto_reg") == "enable",
    )
    return (
        train_images_dir,
    )


def MZ_KohyaSSUseConfig_call(args={}):
    args = args.copy()
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
    train_config_template_dir = args.get("train_config_template_dir", None)
    train_config_template_file = os.path.join(
        train_config_template_dir, train_config_template + ".json")

    # if not os.path.exists(workspace_config_file):
    #     train_config_template_dir = args.get("train_config_template_dir", None)
    #     train_config_template_file = os.path.join(
    #         train_config_template_dir, train_config_template + ".json")
    #     shutil.copy(train_config_template_file, workspace_config_file)

    config = None
    with open(train_config_template_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        config["metadata"]["train_type"] = train_config_template
        ckpt_name = args.get("ckpt_name", "")
        if ckpt_name != "" and ckpt_name is not None:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            config["train_config"]["pretrained_model_name_or_path"] = ckpt_path

        # output_dir
        output_dir = os.path.join(workspace_dir, "output")
        config["train_config"]["output_dir"] = output_dir

        datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # output_name
        config["train_config"]["output_name"] = f"{workspace_name}_{train_config_template}_{datetime}"

        dataset_config_path = os.path.join(
            workspace_dir, "dataset.json")
        if not os.path.exists(dataset_config_path):
            dataset_config_path = os.path.join(
                workspace_dir, "dataset.toml")

        config["train_config"]["dataset_config"] = dataset_config_path

        config["train_config"]["max_train_steps"] = str(
            args.get("max_train_steps"))

        config["train_config"]["max_train_epochs"] = str(
            args.get("max_train_epochs"))
        if config["train_config"]["max_train_epochs"] == "0":
            config["train_config"]["max_train_epochs"] = False

        config["train_config"]["save_every_n_epochs"] = str(
            args.get("save_every_n_epochs"))

        config["train_config"]["learning_rate"] = str(
            args.get("learning_rate"))

        advanced_config = args.get("save_advanced_config", {}).copy()
        if len(advanced_config) == 0:
            advanced_config = args.get("advanced_config", {}).copy()

        for k in advanced_config:
            print(f"{k}= {advanced_config[k]}")

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


def MZ_KohyaSSAdvConfig_call(args={}):
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

    try:
        import diffusers
    except ImportError:
        os.system(f"{sys.executable} -m pip install diffusers")
    try:
        import accelerate
    except ImportError:
        os.system(f"{sys.executable} -m pip install accelerate")


import logging


def generate_dataset_config(workspace_dir, output_path, enable_bucket=True, resolution=512, batch_size=1, image_dir=None, conditioning_data_dir=None, caption_extension=".caption", num_repeats=10, is_reg=False):

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

    if conditioning_data_dir is not None:
        config["datasets"][0]["subsets"][0]["conditioning_data_dir"] = conditioning_data_dir

    if is_reg:
        config["datasets"][0]["subsets"].append({
            "is_reg": True,
            "image_dir": os.path.join(workspace_dir, "reg_images"),
            "num_repeats": 1,
        })

    if output_path.endswith(".toml"):
        check_install()
        import toml
        with open(output_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
    elif output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    else:
        raise Exception(f"不支持的文件格式: {output_path}")


from PIL import Image


def get_sample_images(workspace_dir, output_name):
    sample_images_dir = os.path.join(
        workspace_dir, "sample_images"
    )
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
    if pil_images is None or len(pil_images) == 0:
        return Image.new("RGB", (512, 512), (255, 255, 255))
    result = Utils.xy_image(
        pre_render_images=pil_images,
        pre_render_texts_x=pre_render_texts_x,
        pre_render_texts_y=[""],
    )
    return result


def run_hook_kohya_ss_run_file(workspace_dir, output_name, kohya_ss_tool_dir, trainer_func, use_screen=False):

    train_config_file = os.path.join(workspace_dir, "config.json")

    exec_pyfile = os.path.join(os.path.dirname(
        __file__), "hook_kohya_ss_run.py",)

    is_running = True

    taesd_type = "sd1_5"
    if trainer_func.find("sd1_5") != -1:
        taesd_type = "sd1_5"
    if trainer_func.find("sdxl") != -1:
        taesd_type = "sdxl"
    if trainer_func.find("hunyuan1_1") != -1:
        taesd_type = "sdxl"

    pb = Utils.progress_bar(0, taesd_type)

    import traceback

    import comfy.model_management

    stop_server = None

    def log_callback(log):
        try:
            comfy.model_management.throw_exception_if_processing_interrupted()
        except Exception as e:
            stop_server()
            if process_instance is not None:
                process_instance.stop()  # 强制关闭进程
            return is_running

        try:
            resp = log
            if resp.get("type") == "sample_images":
                global_step = resp.get("global_step")
                xy_img = get_sample_images(workspace_dir, output_name)

                max_side = max(xy_img.width, xy_img.height)
                # print(f"global_step: {global_step}, max_train_steps: {max_train_steps}")

                total_steps = resp.get("total_steps")
                pb.update(
                    int(global_step), int(total_steps), ("JPEG", xy_img, max_side))
            else:
                print(f"LOG: {log}")
        except Exception as e:
            print(f"LOG: {log} e: {e} ")
            print(f"stack: {traceback.format_exc()}")
        return is_running

    stop_server, port = Utils.Simple_Server(log_callback)
    try:
        cmd_list = [sys.executable, exec_pyfile, "--sys_path", kohya_ss_tool_dir,
                    "--config", train_config_file, "--train_func", trainer_func, "--master_port", str(port)]
        startup_script_path_sh = os.path.join(workspace_dir, "start_train.sh")
        startup_script_path_bat = os.path.join(
            workspace_dir, "start_train.bat")
        with open(startup_script_path_sh, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd_list))
        with open(startup_script_path_bat, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd_list))

        from .mz_train_tools_utils import HSubprocess

        process_instance = HSubprocess(
            cmd_list)
        process_instance.wait()

        stop_server()
        is_running = False
    except Exception as e:
        stop_server()
        is_running = False
        raise Exception(f"训练失败!!! 具体报错信息请查看控制台...")


def get_models_type(model_path):

    from comfy.model_detection import model_config_from_unet, unet_prefix_from_state_dict, detect_unet_config
    from comfy.supported_models import SDXL, SD15
    import safetensors.torch
    sd = safetensors.torch.load_file(model_path, device="cpu")
    unet_prefix = unet_prefix_from_state_dict(sd)

    model_type = model_config_from_unet(sd, unet_prefix)

    if isinstance(model_type, SD15):
        return "lora_sd1_5"
    if isinstance(model_type, SDXL):
        return "lora_sdxl"

    raise Exception(f"不支持的模型类型: {model_type}")


def generate_kohya_ss_config(args):
    args = args.copy()
    workspace_config = args.get("workspace_config", {}).copy()
    advanced_config = args.get("advanced_config", {}).copy()
    train_config = args.get("train_config", {}).copy()

    workspace_name = workspace_config.get("workspace_name", None)
    if workspace_name is None or workspace_name == "":
        raise Exception("工作区名称不能为空(workspace_name is required)")
    workspace_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces", workspace_name)

    if not os.path.exists(workspace_dir):
        raise Exception(f"工作区不存在: {workspace_dir}")

    workspace_config_file = os.path.join(workspace_dir, "config.json")

    ckpt_name = args.get("ckpt_name", "")
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

    train_config_template = get_models_type(ckpt_path)

    train_config_template_dir = os.path.join(
        os.path.dirname(__file__), "configs", "kohya_ss_lora")
    train_config_template_file = os.path.join(
        train_config_template_dir, train_config_template + ".json")

    # raise Exception(f"args: {json.dumps(args, indent=4, ensure_ascii=False)}")
    config = None
    with open(train_config_template_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        config["metadata"]["train_type"] = train_config_template
        ckpt_name = args.get("ckpt_name", "")
        if ckpt_name != "" and ckpt_name is not None:
            config["train_config"]["pretrained_model_name_or_path"] = ckpt_path

        # output_dir
        output_dir = os.path.join(workspace_dir, "output")
        config["train_config"]["output_dir"] = output_dir

        datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # output_name
        config["train_config"]["output_name"] = f"{workspace_name}_{train_config_template}_{datetime}"

        dataset_config_path = os.path.join(
            workspace_dir, "dataset.json")
        if not os.path.exists(dataset_config_path):
            dataset_config_path = os.path.join(
                workspace_dir, "dataset.toml")

        config["train_config"]["dataset_config"] = dataset_config_path

        config["train_config"]["max_train_steps"] = str(
            args.get("max_train_steps"))

        config["train_config"]["max_train_epochs"] = str(
            args.get("max_train_epochs"))
        if config["train_config"]["max_train_epochs"] == "0":
            config["train_config"]["max_train_epochs"] = False

        config["train_config"]["save_every_n_epochs"] = str(
            args.get("save_every_n_epochs"))

        config["train_config"]["learning_rate"] = str(
            args.get("learning_rate"))

        for k in advanced_config:
            print(f"{k}= {advanced_config[k]}")

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

    # raise Exception(f"MZ_KohyaSSUseConfig_call: {args}")
    return config


def generate_regularization_folder(ckpt_name, workspace_dir, caption_extension,
                                   width=512, height=512):
    train_images_dir = os.path.join(workspace_dir, "train_images")
    saved_images_path = os.listdir(train_images_dir)

    reg_dir = os.path.join(workspace_dir, "reg_images")

    caption_files = []
    caption_value = []
    for i, filename in enumerate(saved_images_path):
        if filename.lower().endswith(caption_extension):
            caption_files.append(filename)
            with open(os.path.join(train_images_dir, filename), "r", encoding="utf-8") as f:
                caption_value.append(f.read())

    if len(caption_files) == 0:
        return
    os.makedirs(reg_dir, exist_ok=True)
    for filename, caption in zip(caption_files, caption_value):
        if caption == "":
            continue

        with open(os.path.join(reg_dir, filename), "w", encoding="utf-8") as f:
            f.write(caption)

        image_filename = filename.replace(caption_extension, ".webp")
        Utils.comfyui_fast_generate_picture(
            ckpt_name=ckpt_name,
            prompt=caption,
            output=os.path.join(reg_dir, image_filename),
            width=width,
            height=height,
        )


def MZ_KohyaSSTrain_call(args={}):
    args = args.copy()
    workspace_config = args.get("workspace_config").copy()
    base_lora = args.get("base_lora", "empty")
    sample_generate = args.get("sample_generate", "enable")
    sample_prompt = args.get("sample_prompt", "")

    workspace_name = workspace_config.get("workspace_name")
    workspace_dir = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces", workspace_name)

    if not os.path.exists(workspace_dir):
        raise Exception(f"工作区不存在: {workspace_dir}")

    config = generate_kohya_ss_config(args)

    branch_local_name = workspace_config.get(
        "branch_local_name", "kohya_ss_lora")
    kohya_ss_tool_dir = os.path.join(
        Utils.get_minus_zone_models_path(), "train_tools", branch_local_name)

    if kohya_ss_tool_dir not in sys.path:
        sys.path.append(kohya_ss_tool_dir)
    check_install()

    if base_lora == "empty":
        pass
    elif base_lora == "latest":
        workspace_lora_dir = os.path.join(workspace_dir, "output")
        if os.path.exists(workspace_lora_dir):
            workspace_lora_files = Utils.listdir(workspace_lora_dir)
            workspace_lora_files = list(
                filter(lambda x: x.endswith(".safetensors"), workspace_lora_files))
            workspace_lora_files = list(
                map(lambda x: os.path.join(workspace_lora_dir, x), workspace_lora_files))
            # 排序
            workspace_lora_files = sorted(
                workspace_lora_files, key=lambda x: os.path.getctime(x), reverse=True)
            if len(workspace_lora_files) > 0:
                base_lora = os.path.join(
                    workspace_lora_dir, workspace_lora_files[0])
        else:
            base_lora = "empty"
    else:
        pass

    if base_lora != "empty" and os.path.exists(base_lora):
        config["train_config"]["network_weights"] = base_lora
        config["train_config"]["dim_from_weights"] = True

        if "network_dim" in config["train_config"]:
            del config["train_config"]["network_dim"]
        if "network_alpha" in config["train_config"]:
            del config["train_config"]["network_alpha"]
        if "network_dropout" in config["train_config"]:
            del config["train_config"]["network_dropout"]

    base_controlnet = args.get("base_controlnet", "empty")
    if base_controlnet == "empty":
        pass
    elif base_controlnet == "latest":
        workspace_controlnet_dir = os.path.join(workspace_dir, "output")
        if os.path.exists(workspace_controlnet_dir):
            workspace_controlnet_files = Utils.listdir(
                workspace_controlnet_dir)
            workspace_controlnet_files = list(
                filter(lambda x: x.endswith(".safetensors"), workspace_controlnet_files))
            workspace_controlnet_files = list(
                map(lambda x: os.path.join(workspace_controlnet_dir, x), workspace_controlnet_files))
            # 排序
            workspace_controlnet_files = sorted(
                workspace_controlnet_files, key=lambda x: os.path.getctime(x), reverse=True)
            if len(workspace_controlnet_files) > 0:
                base_controlnet = os.path.join(
                    workspace_controlnet_dir, workspace_controlnet_files[0])
        else:
            base_controlnet = "empty"
    else:
        pass

    if base_controlnet != "empty" and os.path.exists(base_controlnet):
        config["train_config"]["controlnet_model_name_or_path"] = base_controlnet

    train_type = config.get("metadata").get("train_type")

    if sample_generate == "enable":
        config["other_config"] = {
            "sample_prompt": sample_prompt,
        }
    else:
        config["other_config"] = {}

    workspace_config_file = os.path.join(workspace_dir, "config.json")
    with open(workspace_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    dataset_config_file = config["train_config"]["dataset_config"]
    with open(dataset_config_file, "r", encoding="utf-8") as f:
        dataset_config = json.load(f)

    if dataset_config is None:
        raise Exception("dataset_config is None")

    subsets_list = dataset_config.get("datasets")[0].get("subsets")
    is_reg = True if len(subsets_list) == 2 and subsets_list[1].get(
        "is_reg") else False
    if is_reg:
        caption_extension = subsets_list[0].get("caption_extension")
        ckpt_name = args.get("ckpt_name")
        resolution = dataset_config.get("datasets")[0].get("resolution")
        if f"{resolution}".find("x") != -1:
            resolutionsp = resolution.split("x")
            width = int(resolutionsp[0])
            height = int(resolutionsp[1])
        elif f"{resolution}".find("*") != -1:
            resolutionsp = resolution.split("*")
            width = int(resolutionsp[0])
            height = int(resolutionsp[1])
        else:
            width = height = int(resolution)

        generate_regularization_folder(
            ckpt_name, workspace_dir, caption_extension, width=width, height=height)

    # raise Exception(
    #     f"config: {json.dumps(dataset_config, indent=4, ensure_ascii=False)}")

    output_name = config["train_config"].get("output_name")

    if train_type == "lora_sd1_5":
        run_hook_kohya_ss_run_file(
            workspace_dir, output_name, kohya_ss_tool_dir, "run_lora_sd1_5")
    elif train_type == "lora_sdxl":
        run_hook_kohya_ss_run_file(
            workspace_dir, output_name, kohya_ss_tool_dir, "run_lora_sdxl")
    elif train_type == "controlnet_sd1_5":

        conditioning_images_dir = os.path.join(
            workspace_dir, "conditioning_images")
        conditioning_images_onec = ""
        if os.path.exists(conditioning_images_dir):
            conditioning_images_onec = Utils.listdir(
                conditioning_images_dir)[0]
            config["other_config"]["controlnet_image"] = os.path.join(
                conditioning_images_dir, conditioning_images_onec)
            with open(workspace_config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

        run_hook_kohya_ss_run_file(
            workspace_dir, output_name, kohya_ss_tool_dir, "run_controlnet_sd1_5")
    else:
        raise Exception(
            f"暂时不支持的训练类型: {train_type}")

    return (
        "训练完成",
    )
