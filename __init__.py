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
                ], {"default": "none"}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("MZ_TT_SS_WorkspaceConfig",)
    RETURN_NAMES = ("workspace_config",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_KohyaSSInitWorkspace_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSInitWorkspace"] = MZ_KohyaSSInitWorkspace
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSInitWorkspace"] = f"{AUTHOR_NAME} - KohyaSSInitWorkspace"


class MZ_ImagesCopyWorkspace:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workspace_config": ("MZ_TT_SS_WorkspaceConfig",),
                "images": ("IMAGE",),
                "enable_bucket": (["enable", "disable"], {"default": "enable"}),
                "resolution": ("INT", {"default": 512}),
                "num_repeats": ("INT", {"default": 1}),
                "batch_size": ("INT", {"default": 1}),
                # "class_name": ("STRING", {"default": "girl", "dynamicPrompts": True}),
                "force_clear": (["enable", "disable"], {"default": "disable"}),
                "force_clear_only_images": (["enable", "disable"], {"default": "disable"}),
                "same_caption_generate": (["enable", "disable"], {"default": "disable"}),
                "same_caption": ("STRING", {"default": "", "dynamicPrompts": True, "multiline": True}),
            },
        }

    RETURN_TYPES = (f"STRING",)
    RETURN_NAMES = ("workspace_images_dir",)

    # OUTPUT_NODE = True

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_ImageSelecter_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ImagesCopyWorkspace"] = MZ_ImagesCopyWorkspace
NODE_DISPLAY_NAME_MAPPINGS["MZ_ImagesCopyWorkspace"] = f"{AUTHOR_NAME} - ImagesCopyWorkspace"


class MZ_KohyaSSUseConfig:
    train_config_template_dir = os.path.join(
        os.path.dirname(__file__), "configs", "kohya_ss_lora"
    )

    @classmethod
    def INPUT_TYPES(s):
        train_config_templates = os.listdir(s.train_config_template_dir)
        # 去掉json后缀
        train_config_templates = [os.path.splitext(x)[0]
                                  for x in train_config_templates]
        return {
            "required": {
                "workspace_config": ("MZ_TT_SS_WorkspaceConfig",),
                "workspace_images_dir": ("STRING", {"forceInput": True}),
                "train_config_template": (train_config_templates,),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "max_train_steps": ("INT", {"default": 300}),
                "max_train_epochs": ("INT", {"default": 0}),
                "save_every_n_epochs": ("INT", {"default": 20}),
                "learning_rate": ("STRING", {"default": "1e-5"}),
            },
            "optional": {
                "save_advanced_config": ("MZ_TT_SS_AdvConfig",),
            }
        }

    RETURN_TYPES = (f"MZ_TT_SS_TrainConfig",)
    RETURN_NAMES = ("train_config",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        kwargs["train_config_template_dir"] = self.train_config_template_dir
        return mz_train_tools_core.MZ_KohyaSSUseConfig_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSUseConfig"] = MZ_KohyaSSUseConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSUseConfig"] = f"{AUTHOR_NAME} - KohyaSSUseConfig"


class MZ_KohyaSSAdvConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "xformers": (["enable", "disable"], {"default": "enable"}),
                "sdpa": (["enable", "disable"], {"default": "disable"}),
                "fp8_base": (["enable", "disable"], {"default": "disable"}),
                "mixed_precision": (["no", "fp16", "bf16"], {"default": "fp16"}),
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
                "shuffle_caption": (["enable", "disable"], {"default": "enable"}),
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

    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        return (kwargs,)


NODE_CLASS_MAPPINGS["MZ_KohyaSSAdvConfig"] = MZ_KohyaSSAdvConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSAdvConfig"] = f"{AUTHOR_NAME} - KohyaSSAdvConfig"


class MZ_KohyaSSTrain:

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

        return {
            "required": {
                "train_config": ("MZ_TT_SS_TrainConfig",),
                "base_lora": (loras, {"default": "latest"}),
                "sample_generate": (["enable", "disable"], {"default": "enable"}),
                "sample_prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_KohyaSSTrain_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSTrain"] = MZ_KohyaSSTrain
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSTrain"] = f"{AUTHOR_NAME} - KohyaSSTrain"


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

    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        from PIL import Image
        images = []
        image_dir = kwargs["directory"]
        if not os.path.exists(image_dir):
            return (images,)
        images = os.listdir(image_dir)
        images = [x for x in images if x.endswith(".png") or x.endswith(".jpg")]
        images = [os.path.join(image_dir, x) for x in images]


        pil_images = []
        for image in images:
            pil_images.append(Image.open(image))

        tensor_images = []
        for pil_image in pil_images:
            tensor_images.append(Utils.pil2tensor(pil_image))

        return (tensor_images,)


NODE_CLASS_MAPPINGS["MZ_LoadImagesFromDirectoryPath"] = MZ_LoadImagesFromDirectoryPath
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_LoadImagesFromDirectoryPath"] = f"{AUTHOR_NAME} - LoadImagesFromDirectoryPath"
