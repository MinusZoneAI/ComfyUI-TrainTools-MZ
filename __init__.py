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
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), )
            },
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
