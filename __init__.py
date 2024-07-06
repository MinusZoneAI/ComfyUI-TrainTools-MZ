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
                "batch_size": ("INT", {"default": 1}),
                "force_clear": (["enable", "disable"], {"default": "disable"}),
                "force_clear_only_images": (["enable", "disable"], {"default": "disable"}),
                "same_caption_generate": (["enable", "disable"], {"default": "disable"}),
                "same_caption": ("STRING", {"default": "", "dynamicPrompts": True, "multiline": True}),
            },
            "optional": {
                "conditioning_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = (f"STRING",)
    RETURN_NAMES = ("workspace_images_dir",)

    # OUTPUT_NODE = True
    MZ_DESC = """
如果训练类型是controlnet,必须传入预处理后的图片(conditioning_images)
"""

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_ImageSelecter_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_ImagesCopyWorkspace"] = MZ_KohyaSSDatasetConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_ImagesCopyWorkspace"] = f"{AUTHOR_NAME} - ImagesCopyWorkspace"

# 别名
NODE_CLASS_MAPPINGS["MZ_KohyaSSDatasetConfig"] = MZ_KohyaSSDatasetConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSDatasetConfig"] = f"{AUTHOR_NAME} - KohyaSSDatasetConfig"


class MZ_KohyaSSUseConfig_oldversion:
    train_config_template_dir = os.path.join(
        os.path.dirname(__file__), "configs", "kohya_ss_lora"
    )

    @classmethod
    def INPUT_TYPES(s):
        train_config_templates = Utils.listdir(s.train_config_template_dir)

        # 去掉json后缀
        train_config_templates = [os.path.splitext(x)[0]
                                  for x in train_config_templates]
        return {
            "required": {
                "workspace_config": ("MZ_TT_SS_WorkspaceConfig",),
                "workspace_images_dir": ("STRING", {"forceInput": True}),
                "train_config_template": (train_config_templates,),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "max_train_steps": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "max_train_epochs": ("INT", {"default": 100, "min": 0, "max": 0x7fffffff}),
                "save_every_n_epochs": ("INT", {"default": 10}),
                "learning_rate": ("STRING", {"default": "1e-5"}),
            },
            "optional": {
                "save_advanced_config": ("MZ_TT_SS_AdvConfig",),
            }
        }

    RETURN_TYPES = (f"MZ_TT_SS_TrainConfig",)
    RETURN_NAMES = ("train_config",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss" + "/v1"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        kwargs["train_config_template_dir"] = self.train_config_template_dir
        return mz_train_tools_core.MZ_KohyaSSUseConfig_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSUseConfig"] = MZ_KohyaSSUseConfig_oldversion
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KohyaSSUseConfig"] = f"{AUTHOR_NAME} - KohyaSSUseConfig(old version)"


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
        return (kwargs,)


NODE_CLASS_MAPPINGS["MZ_KohyaSSAdvConfig"] = MZ_KohyaSSAdvConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSSAdvConfig"] = f"{AUTHOR_NAME} - KohyaSSAdvConfig"


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class MZ_KohyaSSTrain_oldversion:

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

        return {
            "required": {
                "train_config": ("MZ_TT_SS_TrainConfig",),
                "base_lora": (loras, {"default": "latest"}),
                "sample_generate": (["enable", "disable"], {"default": "enable"}),
                "sample_prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
            },
            "optional": {
                "has_no_effect": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss" + "/v1"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        return mz_train_tools_core.MZ_KohyaSSTrain_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSTrain"] = MZ_KohyaSSTrain_oldversion
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KohyaSSTrain"] = f"{AUTHOR_NAME} - KohyaSSTrain(old version)"


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

        # 去掉json后缀
        train_config_templates = [os.path.splitext(x)[0]
                                  for x in train_config_templates]
        return {
            "required": {
                "workspace_config": ("MZ_TT_SS_WorkspaceConfig",),
                "train_config_template": (train_config_templates,),
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

        train_config = {
            "workspace_config": kwargs["workspace_config"],
            "train_config_template": kwargs["train_config_template"],
            "ckpt_name": kwargs["ckpt_name"],
            "max_train_steps": kwargs["max_train_steps"],
            "max_train_epochs": kwargs["max_train_epochs"],
            "save_every_n_epochs": kwargs["save_every_n_epochs"],
            "learning_rate": kwargs["learning_rate"],
        }

        train_config["train_config_template_dir"] = self.train_config_template_dir

        advanced_config = kwargs.get("advanced_config", None)

        if advanced_config is not None:
            for k, v in advanced_config.items():
                train_config[k] = v

        kwargs["train_config"] = train_config
        return mz_train_tools_core.MZ_KohyaSSTrain_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSSLoraTrain"] = MZ_KohyaSSLoraTrain
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KohyaSSLoraTrain"] = f"{AUTHOR_NAME} - KohyaSSTrain(lora)"


class MZ_KohyaSSControlnetTrain:

    @classmethod
    def INPUT_TYPES(s):
        models = [
            "latest",
            "empty",
        ]

        comfyui_full_m_path = []
        comfyui_basemodels = folder_paths.get_filename_list("controlnet")
        for b_model in comfyui_basemodels:
            m_path = folder_paths.get_full_path("controlnet", b_model)
            comfyui_full_m_path.append(m_path)

        # 按创建时间排序
        comfyui_full_m_path = sorted(
            comfyui_full_m_path, key=lambda x: os.path.getctime(x), reverse=True)

        models = models + comfyui_full_m_path

        return {
            "required": {
                "train_config": ("MZ_TT_SS_TrainConfig",),
                "base_controlnet": (models, {"default": "latest"}),
                "sample_generate": (["enable", "disable"], {"default": "enable"}),
                "sample_prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
            },
            "optional": {
                "has_no_effect": (AlwaysEqualProxy("*"),),
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


NODE_CLASS_MAPPINGS["MZ_KohyaSSControlnetTrain"] = MZ_KohyaSSControlnetTrain
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KohyaSSControlnetTrain"] = f"{AUTHOR_NAME} - KohyaSSTrain(controlnet)"


class MZ_KohyaSS_KohakuBlueleaf_HYHiDLoraTrain:
    KohyaSSLoraTrain = MZ_KohyaSSLoraTrain()

    @classmethod
    def INPUT_TYPES(s):
        models, folders, vae_models, unet_models, loras = HYDiT_paths()
        _kohya_input_types = s.KohyaSSLoraTrain.INPUT_TYPES()

        # "workspace_config": kwargs["workspace_config"],
        # "train_config_template": kwargs["train_config_template"],
        # "ckpt_name": kwargs["ckpt_name"],
        # "max_train_steps": kwargs["max_train_steps"],
        # "max_train_epochs": kwargs["max_train_epochs"],
        # "save_every_n_epochs": kwargs["save_every_n_epochs"],
        # "learning_rate": kwargs["learning_rate"],

        kohya_input_types = {"required": {}, "optional": {}}
        kohya_input_types["required"]["unet_path"] = (
            ["auto"] + models + unet_models, {"default": "auto"})
        kohya_input_types["required"]["vae_ema_path"] = (
            ["auto"] + folders + vae_models, {"default": "auto"})
        kohya_input_types["required"]["text_encoder_path"] = (
            ["auto"] + folders, {"default": "auto"})
        kohya_input_types["required"]["tokenizer_path"] = (
            ["auto"] + folders, {"default": "auto"})
        kohya_input_types["required"]["t5_encoder_path"] = (
            ["none", "auto"] + folders, {"default": "none"})

        for k, v in _kohya_input_types["required"].items():
            if k == "ckpt_name":
                continue
            if k == "ema_to_module":
                continue
            kohya_input_types["required"][k] = v

        for k, v in _kohya_input_types["optional"].items():
            kohya_input_types["optional"][k] = v

        return kohya_input_types

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss_HYDiT_kohakublueleaf"

    def start(self, **kwargs):
        kwargs["hunyuan_models_config"] = {
            "unet_path": kwargs["unet_path"],
            "vae_ema_path": kwargs["vae_ema_path"],
            "text_encoder_path": kwargs["text_encoder_path"],
            "tokenizer_path": kwargs["tokenizer_path"],
            "t5_encoder_path": kwargs["t5_encoder_path"],
        }
        kwargs["ckpt_name"] = None
        return self.KohyaSSLoraTrain.start(**kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSS_KohakuBlueleaf_HYHiDLoraTrain"] = MZ_KohyaSS_KohakuBlueleaf_HYHiDLoraTrain
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KohyaSS_KohakuBlueleaf_HYHiDLoraTrain"] = f"{AUTHOR_NAME} - KohyaSS_KohakuBlueleaf_HYHiDLoraTrain"


class MZ_KohyaSS_KohakuBlueleaf_HYHiDInitWorkspace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": ("STRING", {"default": ""}),
                "branch": ("STRING", {"default": "0dc79edc01f2000de1dad5ad6d20d8b099bfafe2"}),
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
    CATEGORY = CATEGORY_NAME + "/kohya_ss_HYDiT_kohakublueleaf"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        kwargs["branch_repoid"] = "KohakuBlueleaf/sd-scripts"
        kwargs["branch_local_name"] = "KohakuBlueleaf_kohya_ss_lora"
        return mz_train_tools_core.MZ_KohyaSSInitWorkspace_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSS_KohakuBlueleaf_HYHiDInitWorkspace"] = MZ_KohyaSS_KohakuBlueleaf_HYHiDInitWorkspace
NODE_DISPLAY_NAME_MAPPINGS[
    "MZ_KohyaSS_KohakuBlueleaf_HYHiDInitWorkspace"] = f"{AUTHOR_NAME} - KohyaSS_KohakuBlueleaf_HYHiDInitWorkspace"


class MZ_KohyaSS_KohakuBlueleaf_HYHiDSimpleT2I:
    @classmethod
    def INPUT_TYPES(s):
        models, folders, vae_models, unet_models, _ = HYDiT_paths()
        comfyui_full_loras = []
        comfyui_loras = folder_paths.get_filename_list("loras")
        for lora in comfyui_loras:
            lora_path = folder_paths.get_full_path("loras", lora)
            comfyui_full_loras.append(lora_path)
        return {
            "required": {
                "branch": ("STRING", {"default": "0dc79edc01f2000de1dad5ad6d20d8b099bfafe2"}),
                "source": ([
                    "github",
                    "githubfast",
                    "521github",
                    "kkgithub",
                ], {"default": "github"}),
                "version": (["1.1", "1.2"], {"default": "1.2"}),
                "unet_path": (["auto"] + models + unet_models, {"default": "auto"}),
                "vae_ema_path": (["auto"] + folders + vae_models, {"default": "auto"}),
                "text_encoder_path": (["auto"] + folders, {"default": "auto"}),
                "tokenizer_path": (["auto"] + folders, {"default": "auto"}),
                "t5_encoder_path": (["none", "auto"] + folders, {"default": "none"}),
                "lora_path": (["none"] + comfyui_full_loras, {"default": "none"}), 
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "scheduler": ([
                    "euler_ancestral", "dpmpp_2m_sde"
                ], {"default": "dpmpp_2m_sde"}),
                "prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
                "negative_prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
                "width": ("INT", {"default": 1024, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "max": 8192, "step": 16}),
                "keep_device": (["enable", "disable"], {"default": "enable"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "start"
    CATEGORY = CATEGORY_NAME + "/kohya_ss_HYDiT_kohakublueleaf"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core)
        kwargs["branch_repoid"] = "KohakuBlueleaf/sd-scripts"
        kwargs["branch_local_name"] = "KohakuBlueleaf_kohya_ss_lora"
        return mz_train_tools_core.MZ_KohyaSS_KohakuBlueleaf_HYHiDSimpleT2I_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_KohyaSS_KohakuBlueleaf_HYHiDSimpleT2I"] = MZ_KohyaSS_KohakuBlueleaf_HYHiDSimpleT2I
NODE_DISPLAY_NAME_MAPPINGS["MZ_KohyaSS_KohakuBlueleaf_HYHiDSimpleT2I"] = f"{AUTHOR_NAME} - KohyaSS_KohakuBlueleaf_HYHiDSimpleT2I"


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
            ".png") or x.lower().endswith(".jpg")]
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


from . import mz_train_tools_core_HYDiT


class MZ_HYDiTInitWorkspace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "train_name": ("STRING", {"default": ""}),
                "branch": ("STRING", {"default": "5657364143e44ac90f72aeb47b81bd505a95665d"}),
                "source": ([
                    "github",
                    "githubfast",
                    "521github",
                    "kkgithub",
                ], {"default": "github"}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("MZ_TT_HYDiT_WorkspaceConfig",)
    RETURN_NAMES = ("workspace_config",)

    FUNCTION = "start"

    CATEGORY = CATEGORY_NAME + "/HYDiT_native"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core_HYDiT)
        return mz_train_tools_core_HYDiT.MZ_HYDiTInitWorkspace_call(kwargs.copy())


NODE_CLASS_MAPPINGS["MZ_HYDiTInitWorkspace"] = MZ_HYDiTInitWorkspace
NODE_DISPLAY_NAME_MAPPINGS["MZ_HYDiTInitWorkspace"] = f"{AUTHOR_NAME} - HYDiTInitWorkspace"


class MZ_HYDiTDatasetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workspace_config": ("MZ_TT_HYDiT_WorkspaceConfig",),
                "resolution": ("INT", {"default": 1024}),
                "force_clear": (["enable", "disable"], {"default": "disable"}),
                "force_clear_only_images": (["enable", "disable"], {"default": "disable"}),
                "same_caption_generate": (["enable", "disable"], {"default": "disable"}),
                "same_caption": ("STRING", {"default": "", "dynamicPrompts": True, "multiline": True}),
            },
            "optional": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("workspace_images_dir",)

    FUNCTION = "start"

    CATEGORY = CATEGORY_NAME + "/HYDiT_native"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core_HYDiT)
        return mz_train_tools_core_HYDiT.MZ_HYDiTDatasetConfig_call(kwargs.copy())


NODE_CLASS_MAPPINGS["MZ_HYDiTDatasetConfig"] = MZ_HYDiTDatasetConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_HYDiTDatasetConfig"] = f"{AUTHOR_NAME} - HYDiTDatasetConfig"


class MZ_HYDiTAdvConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lr": ("STRING", {"default": "1e-5"}),
                "rope_real": (["enable", "disable"], {"default": "enable"}),
                # ['Wqkv', 'q_proj', 'kv_proj', 'out_proj']
                "target_modules_Wqkv": (["enable", "disable"], {"default": "enable"}),
                "target_modules_q_proj": (["enable", "disable"], {"default": "enable"}),
                "target_modules_kv_proj": (["enable", "disable"], {"default": "enable"}),
                "target_modules_out_proj": (["enable", "disable"], {"default": "enable"}),
                "warmup_min_lr": ("STRING", {"default": "1e-6"}),
                # parser.add_argument("--warmup-num-steps", type=float, default=0)
                # parser.add_argument("--weight-decay", type=float, default=0, help="weight-decay in optimizer")
                "weight_decay": ("FLOAT", {"default": 0}),
                "warmup_num_steps": ("FLOAT", {"default": 0}),
                # parser.add_argument("--uncond-p", type=float, default=0.2,
                #                     help="The probability of dropping training text used for CLIP feature extraction")
                # parser.add_argument("--uncond-p-t5", type=float, default=0.2,
                #                     help="The probability of dropping training text used for mT5 feature extraction")
                "uncond_p": ("FLOAT", {"default": 0.2}),
                "uncond_p_t5": ("FLOAT", {"default": 0.2}),

                # parser.add_argument("--use-flash-attn", action="store_true", help="During training, "
                #                                                                 "flash attention is used to accelerate training.")
                # parser.add_argument("--no-flash-attn", dest="use_flash_attn",
                #                     action="store_false", help="During training, flash attention is not used to accelerate training.")
                # parser.add_argument("--use-zero-stage", type=int, default=1, help="Use AngelPTM zero stage. Support 2 and 3")
                # parser.add_argument("--grad-accu-steps", type=int, default=1, help="Gradient accumulation steps.")
                "use_flash_attn": (["enable", "disable"], {"default": "disable"}),
                "use_zero_stage": ("INT", {"default": 2}),
                "grad_accu_steps": ("INT", {"default": 1}),
                #  parser.add_argument("--extra-fp16", action="store_true", help="Use extra fp16 for vae and text_encoder.")
                "extra_fp16": (["enable", "disable"], {"default": "enable"}),
                # parser.add_argument("--qk-norm", action="store_true", help="Query Key normalization. See http://arxiv.org/abs/2302.05442 for details.")
                "qk_norm": (["enable", "disable"], {"default": "enable"}),
                # parser.add_argument("--norm", type=str, choices=["rms", "laryer"], default="layer", help="Normalization layer type")
                "norm": (["rms", "layer"], {"default": "layer"}),
            }
        }

    RETURN_TYPES = ("MZ_TT_HYDiT_AdvConfig",)
    RETURN_NAMES = ("advanced_config",)

    FUNCTION = "start"

    CATEGORY = CATEGORY_NAME + "/HYDiT_native"

    def start(self, **kwargs):
        return (kwargs.copy(),)


NODE_CLASS_MAPPINGS["MZ_HYDiTAdvConfig"] = MZ_HYDiTAdvConfig
NODE_DISPLAY_NAME_MAPPINGS["MZ_HYDiTAdvConfig"] = f"{AUTHOR_NAME} - HYDiTAdvConfig"


def HYDiT_paths():
    hunyuan_models_path = os.path.join(
        Utils.get_comfyui_models_path(), "hunyuan")
    os.makedirs(hunyuan_models_path, exist_ok=True)

    models = Utils.get_models_by_folder(hunyuan_models_path)

    folders = Utils.get_folders_by_folder(hunyuan_models_path)

    vae_models = Utils.get_models_by_folder(
        os.path.join(Utils.get_comfyui_models_path(), "vae"))
    unet_models = Utils.get_models_by_folder(
        os.path.join(Utils.get_comfyui_models_path(), "unet"))

    workspaces_root = os.path.join(
        folder_paths.output_directory, "mz_train_workspaces")

    loras = mz_train_tools_core_HYDiT.search_loras([
        workspaces_root,
        os.path.join(Utils.get_comfyui_models_path(), "loras"),
    ])

    return models, folders, vae_models, unet_models, loras


class MZ_HYDiTTrain:
    @classmethod
    def INPUT_TYPES(s):

        models, folders, vae_models, unet_models, loras = HYDiT_paths()

        return {
            "required": {
                "workspace_config": ("MZ_TT_HYDiT_WorkspaceConfig",),
                "unet_path": (["auto"] + models + unet_models, {"default": "auto"}),
                "ema_to_module": (["enable", "disable"], {"default": "enable"}),
                "vae_ema_path": (["auto"] + folders + vae_models, {"default": "auto"}),
                "text_encoder_path": (["auto"] + folders, {"default": "auto"}),
                "tokenizer_path": (["auto"] + folders, {"default": "auto"}),
                "t5_encoder_path": (["none", "auto"] + folders, {"default": "none"}),
                "resolution": ("INT", {"default": 1024, "step": 16}),
                "batch_size": ("INT", {"default": 1}),
                "epochs": ("INT", {"default": 50}),
                "ckpt_every": ("INT", {"default": 500}),
                "rank": ("INT", {"default": 8}),
                "base_lora": (["latest", "empty"] + loras, {"default": "latest"}),
                "sample_generate": (["enable", "disable"], {"default": "enable"}),
                "sample_prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
            },
            "optional": {
                "advanced_config": ("MZ_TT_HYDiT_AdvConfig",),
                "workspace_images_dir": ("STRING", {"forceInput": True}),
                "has_no_effect": (AlwaysEqualProxy("*"),),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "start"

    MZ_DESC = """
base_size The base resolution (n, n) from which to create multiple resolutions | Recommended values: 256/512/1024
"""

    OUTPUT_NODE = True

    CATEGORY = CATEGORY_NAME + "/HYDiT_native"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core_HYDiT)
        return mz_train_tools_core_HYDiT.MZ_HYDiTTrain_call(kwargs.copy())


NODE_CLASS_MAPPINGS["MZ_HYDiTTrain"] = MZ_HYDiTTrain
NODE_DISPLAY_NAME_MAPPINGS["MZ_HYDiTTrain"] = f"{AUTHOR_NAME} - HYDiTTrain"


class MZ_HYDiTSimpleT2I:
    @classmethod
    def INPUT_TYPES(s):
        hunyuan_models_path = os.path.join(
            Utils.get_comfyui_models_path(), "hunyuan")
        os.makedirs(hunyuan_models_path, exist_ok=True)

        models = Utils.get_models_by_folder(hunyuan_models_path)
        folders = Utils.get_folders_by_folder(hunyuan_models_path)

        vae_models = Utils.get_models_by_folder(
            os.path.join(Utils.get_comfyui_models_path(), "vae"))
        unet_models = Utils.get_models_by_folder(
            os.path.join(Utils.get_comfyui_models_path(), "unet"))

        comfyui_full_loras = mz_train_tools_core_HYDiT.search_loras([
            os.path.join(Utils.get_comfyui_models_path(), "loras"),
        ])

        return {
            "required": {
                "branch": ("STRING", {"default": "5657364143e44ac90f72aeb47b81bd505a95665d"}),
                "source": ([
                    "github",
                    "githubfast",
                    "521github",
                    "kkgithub",
                ], {"default": "github"}),
                "unet_path": (["auto"] + models + unet_models, {"default": "auto"}),
                "vae_ema_path": (["auto"] + folders + vae_models, {"default": "auto"}),
                "text_encoder_path": (["auto"] + folders, {"default": "auto"}),
                "tokenizer_path": (["auto"] + folders, {"default": "auto"}),
                "t5_encoder_path": (["none", "auto"] + folders, {"default": "auto"}),
                "lora_path": (["none"] + comfyui_full_loras, {"default": "none"}),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "scheduler": ([
                    "ddpm", "ddim", "dpmms", "uni_pc", "dpmpp_2m_karras"
                ], {"default": "ddpm"}),
                "prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
                "negative_prompt": ("STRING", {"default:": "", "dynamicPrompts": True, "multiline": True}),
                "width": ("INT", {"default": 512, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 512, "max": 8192, "step": 16}),
                "keep_device": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    RETURN_NAMES = ("images",)

    FUNCTION = "start"

    CATEGORY = CATEGORY_NAME + "/HYDiT_native"

    def start(self, **kwargs):
        importlib.reload(mz_train_tools_core_HYDiT)
        return mz_train_tools_core_HYDiT.MZ_HYDiTSimpleT2I_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_HYDiTSimpleT2I"] = MZ_HYDiTSimpleT2I
NODE_DISPLAY_NAME_MAPPINGS["MZ_HYDiTSimpleT2I"] = f"{AUTHOR_NAME} - HYDiTSimpleT2I"


class MZ_TrainToolsDebug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "object": (AlwaysEqualProxy("*"),),
                "indent": ("INT", {"default": 2}),
                "depth": ("INT", {"default": 5}),
                "width": ("INT", {"default": 80}),
                "compact": (["enable", "disable"], {"default": "enable"}),
                "sort_keys": (["enable", "disable"], {"default": "enable"}),
                "underscore_numbers": (["enable", "disable"], {"default": "enable"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)

    FUNCTION = "start"

    CATEGORY = CATEGORY_NAME

    def start(self, **kwargs):

        from pprint import pprint, pp
        object = kwargs["object"]
        indent = kwargs["indent"]
        depth = kwargs["depth"]
        width = kwargs["width"]
        compact = kwargs["compact"] == "enable"
        sort_keys = kwargs["sort_keys"] == "enable"
        underscore_numbers = kwargs["underscore_numbers"] == "enable"

        debug = pp(object, stream=None, indent=indent, depth=depth, width=width,
                   compact=compact, sort_dicts=sort_keys, underscore_numbers=underscore_numbers)

        return (debug,)


# NODE_CLASS_MAPPINGS["MZ_TrainToolsDebug"] = MZ_TrainToolsDebug
# NODE_DISPLAY_NAME_MAPPINGS["MZ_TrainToolsDebug"] = f"{AUTHOR_NAME} - TrainToolsDebug"
