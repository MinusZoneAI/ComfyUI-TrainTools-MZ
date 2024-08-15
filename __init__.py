import inspect
import json
import os
import folder_paths
import importlib
from .mz_train_tools_utils import Utils
from . import mz_train_tools_core

NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

AUTHOR_NAME = "MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - TrainTools"


class MZ_KohyaSSInitWorkspace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": ("STRING", {"default": ""}),
                "branch": ("STRING", {"default": "71e2c91330a9d866ec05cdd10584bbb962896a99"}),
                "source": ([
                    "github",
                    "githubfast",
                    "521github",
                    "kkgithub",
                ], {"default": "github"}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("MZ_TT_SS_WorkspaceConfig",)
    RETURN_NAMES = ("workspace_config",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_KohyaSSInitWorkspace_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSInitWorkspace"] = MZ_KohyaSSInitWorkspace
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSInitWorkspace"] = f"{AUTHOR_NAME} - KohyaSSInitWorkspace"


class MZ_KohyaSSDatasetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workspace_config": ("MZ_TT_SS_WorkspaceConfig",),
                "images": ("IMAGE",),
                "enable_bucket": (["enable", "disable"], {"default": "enable"}),
                "resolution": ("INT", {"default": 1024}),
                "num_repeats": ("INT", {"default": 1}),
                "caption_extension": ([".caption", ".txt"], {"default": ".caption"}),
                "batch_size": ("INT", {"default": 1}),
                "force_clear": (["enable", "disable"], {"default": "disable"}),
                "force_clear_only_images": (["enable", "disable"], {"default": "disable"}),
                "same_caption_generate": (["enable", "disable"], {"default": "disable"}),
                "same_caption": ("STRING", {"default": "", "dynamicPrompts": True, "multiline": True}),
                "image_format": (["png", "jpg", "webp"], {"default": "webp"}),
                "dataset_config_extension": ([".toml", ".json"], {"default": ".json"}),
                "auto_reg": (["enable", "disable"], {"default": "enable"}),
            },
            "optional": {
                "conditioning_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = (f"STRING",)
    RETURN_NAMES = ("workspace_images_dir",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_KohyaSSDatasetConfig_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ImagesCopyWorkspace"] = MZ_KohyaSSDatasetConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_ImagesCopyWorkspace"] = f"{AUTHOR_NAME} - ImagesCopyWorkspace"

# 别名
NODE_CLASS_MAPPINGS["MZ_KohyaSSDatasetConfig"] = MZ_KohyaSSDatasetConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSDatasetConfig"] = f"{AUTHOR_NAME} - KohyaSSDatasetConfig"


class MZ_KohyaSSAdvConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "xformers": (["enable", "disable"], {"default": "enable"}),
                "sdpa": (["enable", "disable"], {"default": "disable"}),
                "fp8_base": (["enable", "disable"], {"default": "disable"}),
                "mixed_precision": (["no", "fp16", "bf16"], {"default": "fp16"}),
                "gradient_accumulation_steps": ("INT", {"default": 1}),
                "gradient_checkpointing": (["enable", "disable"], {"default": "disable"}),
                "cache_latents": (["enable", "disable"], {"default": "enable"}),
                "cache_latents_to_disk": (["enable", "disable"], {"default": "enable"}),
                "network_dim": ("INT", {"default": 16}),
                "network_alpha": ("INT", {"default": 8}),
                "network_module": ([
                    "networks.lora",
                    "networks.dylora",
                    "networks.oft",
                ], {"default": "networks.lora"}),
                "network_train_unet_only": (["enable", "disable"], {"default": "enable"}),
                # linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor
                "lr_scheduler": ([
                    "linear",
                    "cosine",
                    "cosine_with_restarts",
                    "polynomial",
                    "constant",
                    "constant_with_warmup",
                    "adafactor",
                ], {"default": "cosine"}),
                "lr_scheduler_num_cycles": ("INT", {"default": 1}),
                # AdamW (default), AdamW8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, AdaFactor
                "optimizer_type": ([
                    "AdamW",
                    "AdamW8bit",
                    "PagedAdamW",
                    "PagedAdamW8bit",
                    "PagedAdamW32bit",
                    "Lion8bit",
                    "PagedLion8bit",
                    "Lion",
                    "SGDNesterov",
                    "SGDNesterov8bit",
                    "DAdaptation",
                    "DAdaptAdaGrad",
                    "DAdaptAdam",
                    "DAdaptAdan",
                    "DAdaptAdanIP",
                    "DAdaptLion",
                    "DAdaptSGD",
                    "AdaFactor",
                ], {"default": "AdamW"}),
                "lr_warmup_steps": ("INT", {"default": 0}),
                "unet_lr": ("STRING", {"default": ""}),
                "text_encoder_lr": ("STRING", {"default": ""}),
                "shuffle_caption": (["enable", "disable"], {"default": "disable"}),
                "save_precision": (["float", "fp16", "bf16"], {"default": "fp16"}),
                "persistent_data_loader_workers": (["enable", "disable"], {"default": "enable"}),
                "no_metadata": (["enable", "disable"], {"default": "enable"}),
                "noise_offset": ("FLOAT", {"default": 0.1}),
                "no_half_vae": (["enable", "disable"], {"default": "enable"}),
                "lowram": (["enable", "disable"], {"default": "disable"}),
            },
        }

    RETURN_TYPES = ("MZ_TT_SS_AdvConfig",)
    RETURN_NAMES = ("advanced_config",)

    FUNCTION = "start"

    CATEGORY = CATEGORY_NAME + "/kohya_ss"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_KohyaSSAdvConfig_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSAdvConfig"] = MZ_KohyaSSAdvConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSAdvConfig"] = f"{AUTHOR_NAME} - KohyaSSAdvConfig"


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class MZ_KohyaSSLoraTrain:
    train_config_template_dir = os.path.join(
        os.path.dirname(__file__), "configs", "kohya_ss_lora"
    )

    @classmethod
    def INPUT_TYPES(s):
        loras = [
            "latest",
            "empty",
        ]

        workspaces_dir = os.path.join(
            folder_paths.output_directory, "mz_train_workspaces")

        # 使用walk查询所有的workspace中的所有lora模型,lora存放在每个workspace的output目录下
        workspaces_loras = []
        for root, dirs, files in os.walk(workspaces_dir):

            # 排除隐藏文件夹
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if root.endswith("output"):
                for file in files:
                    if file.endswith(".safetensors"):
                        workspaces_loras.append(
                            os.path.join(root, file)
                        )

        # 按创建时间排序
        workspaces_loras = sorted(
            workspaces_loras, key=lambda x: os.path.getctime(x), reverse=True)

        comfyui_full_loras = []
        comfyui_loras = folder_paths.get_filename_list("loras")
        for lora in comfyui_loras:
            lora_path = folder_paths.get_full_path("loras", lora)
            comfyui_full_loras.append(lora_path)

        # 按创建时间排序
        comfyui_full_loras = sorted(
            comfyui_full_loras, key=lambda x: os.path.getctime(x), reverse=True)

        loras = loras + workspaces_loras + comfyui_full_loras

        train_config_templates = Utils.listdir(s.train_config_template_dir)

        priority = [
            "lora",
            "1_2"
            "1_1"
        ]
        # 去掉json后缀
        train_config_templates = [os.path.splitext(x)[0]
                                  for x in train_config_templates]

        def priority_sort(x):
            for p in priority:
                if x.find(p) != -1:
                    return priority.index(p)
            return 999

        train_config_templates = sorted(
            train_config_templates, key=priority_sort)
        return {
            "required": {
                "workspace_config": ("MZ_TT_SS_WorkspaceConfig",),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "max_train_steps": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "max_train_epochs": ("INT", {"default": 100, "min": 0, "max": 0x7fffffff}),
                "save_every_n_epochs": ("INT", {"default": 10}),
                "learning_rate": ("STRING", {"default": "1e-5"}),

                "base_lora": (loras, {"default": "latest"}),
                "sample_generate": (["enable", "disable"], {"default": "enable"}),
                "sample_prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
            },
            "optional": {
                "advanced_config": ("MZ_TT_SS_AdvConfig",),
                "caption_completed_flag": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)

        return mz_train_tools_core.MZ_KohyaSSTrain_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSLoraTrain"] = MZ_KohyaSSLoraTrain
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KohyaSSLoraTrain"] = f"{AUTHOR_NAME} - KohyaSSTrain(lora)"


class MZ_LoadImagesFromDirectoryPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/images"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "start"

    CATEGORY = CATEGORY_NAME + "/tools"

    def start(self, **kwargs):
        from PIL import Image
        images = []
        image_dir = kwargs["directory"]
        if not os.path.exists(image_dir):
            return (images,)
        images = Utils.listdir(image_dir)

        images = [x for x in images if x.lower().endswith(
            ".png") or x.lower().endswith(".jpg") or file.lower().endswith(".webp")]
        images = [os.path.join(image_dir, x) for x in images]

        pil_images = []
        for image in images:
            pil_images.append(Image.open(image))

        tensor_images = []
        for pil_image in pil_images:
            tensor_images.append(Utils.pil2tensor(pil_image))

        return (Utils.list_tensor2tensor(tensor_images),)


NODE_CLASS_MAPPINGS["MZ_LoadImagesFromDirectoryPath"] = MZ_LoadImagesFromDirectoryPath
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LoadImagesFromDirectoryPath"] = f"{AUTHOR_NAME} - LoadImagesFromDirectoryPath"
