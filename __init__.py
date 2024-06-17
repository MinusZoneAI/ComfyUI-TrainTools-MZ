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
                "workspace_name": ("STRING", {"default": ""}),
                "branch": ("STRING", {"default": "main"}),
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
                "advanced_config": ("MZ_TT_SS_AdvConfig",),
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
        return {
            "required": {
                "train_config": ("MZ_TT_SS_TrainConfig",),
            },
        }

    RETURN_TYPES = (f"STRING",)
    RETURN_NAMES = ("train_output_dir",)
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_KohyaSSTrain_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSTrain"] = MZ_KohyaSSTrain
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSTrain"] = f"{AUTHOR_NAME} - KohyaSSTrain"
