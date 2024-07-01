import json
import os
from collections import OrderedDict
from os import PathLike
import shutil
from typing import Final

import torch
import torch.nn as nn
from transformers import (
    SiglipImageProcessor,
    SiglipVisionConfig,
    SiglipVisionModel,
    logging,
)
from transformers.image_processing_utils import BatchFeature
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

logging.set_verbosity_error()

URL: Final[str] = (
    "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/aesthetic_predictor_v2_5.pth"
)


class AestheticPredictorV2_5Head(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.scoring_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.scoring_head(image_embeds)


class AestheticPredictorV2_5Model(SiglipVisionModel):
    PATCH_SIZE = 14

    def __init__(self, config: SiglipVisionConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.layers = AestheticPredictorV2_5Head(config)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | ImageClassifierOutputWithNoAttention:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = super().forward(
            pixel_values=pixel_values,
            return_dict=return_dict,
        )
        image_embeds = outputs.pooler_output
        image_embeds_norm = image_embeds / \
            image_embeds.norm(dim=-1, keepdim=True)
        prediction = self.layers(image_embeds_norm)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct()

        if not return_dict:
            return (loss, prediction, image_embeds)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=prediction,
            hidden_states=image_embeds,
        )


class AestheticPredictorV2_5Processor(SiglipImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> BatchFeature:
        return super().__call__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str
        | PathLike = "google/siglip-so400m-patch14-384",
        *args,
        **kwargs,
    ) -> "AestheticPredictorV2_5Processor":
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


def convert_v2_5_from_siglip(
    predictor_name_or_path: str | PathLike | None = None,
    encoder_model_name: str = "google/siglip-so400m-patch14-384",
    *args,
    **kwargs,
) -> tuple[AestheticPredictorV2_5Model, AestheticPredictorV2_5Processor]:
    model = AestheticPredictorV2_5Model.from_pretrained(
        encoder_model_name, *args, **kwargs
    )

    processor = AestheticPredictorV2_5Processor.from_pretrained(
        encoder_model_name, *args, **kwargs
    )

    if predictor_name_or_path is None or not os.path.exists(predictor_name_or_path):
        state_dict = torch.hub.load_state_dict_from_url(
            URL, map_location="cpu")
    else:
        state_dict = torch.load(predictor_name_or_path, map_location="cpu")

    assert isinstance(state_dict, OrderedDict)

    model.layers.load_state_dict(state_dict)
    model.eval()

    return model, processor


from PIL import Image


class AestheticPredictor:
    def __init__(self, clip_path, aesthetic_predictor_path):
        # load model and preprocessor
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            encoder_model_name=clip_path,
            predictor_name_or_path=aesthetic_predictor_path,
        )
        if torch.cuda.is_available():
            self.model = self.model.to(torch.bfloat16).cuda()

    def inference(self, img_paths) -> float:
        if not isinstance(img_paths, list):
            img_paths = [img_paths]

        pil_images = []
        for img_path in img_paths:
            pil_image = Image.open(img_path)
            pil_images.append(pil_image)

        # preprocess image
        pixel_values = self.preprocessor(
            images=pil_images, return_tensors="pt"
        ).pixel_values

        if torch.cuda.is_available():
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # predict aesthetic score
        with torch.inference_mode():
            score = self.model(
                pixel_values).logits.squeeze().float().cpu().numpy()
        prediction_score = score.flatten()

        images_score = {
            img_path: float(score) for img_path, score in zip(img_paths, prediction_score)
        }

        # 按value排序
        images_score = dict(
            sorted(images_score.items(), key=lambda x: x[1], reverse=True))

        return images_score



import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default="")
    parser.add_argument("-o", "--output_dir", type=str, default="")
    parser.add_argument("-c", "--clip_path", type=str,
                        default="D:\\数据集\\siglip-so400m-patch14-384")
    parser.add_argument("-m", "--aes_model", type=str,
                        default="D:\\数据集\\aesthetic_predictor_v2_5.pth")
    parser.add_argument("-s", "--score", type=float, default=5.9)

    parser.add_argument("-b", "--batch_size", type=int, default=1)

    args = parser.parse_args()
    limitscore = args.score

    aes = AestheticPredictor(args.clip_path, args.aes_model)

    input_dir = args.input_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    total = 0
    for root, dirs, files in os.walk(input_dir):
        
        # 排除隐藏文件夹    
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if not file.endswith(".png") and not file.endswith(".jpg"):
                continue
            total += 1

    import tqdm
    print(f"total: {total}")

    images = []

    with tqdm.tqdm(total=total) as pbar:
        for root, dirs, files in os.walk(input_dir):
            # 排除隐藏文件夹    
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if not file.endswith(".png") and not file.endswith(".jpg"):
                    continue
                file_path = os.path.join(root, file)
                pbar.update(1)

                pbar.set_description(
                    f"Processing {os.path.basename(file_path)}")

                if os.path.exists(os.path.join(output_dir, os.path.basename(file_path))):
                    continue

                if len(images) < batch_size:
                    images.append(file_path)
                else:
                    image_scroe_map = aes.inference(images)
                    images = []
                    for img_path, score in image_scroe_map.items():
                        print(f"{img_path}: {score}")

                        if score > limitscore:
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            shutil.copy(img_path, output_dir)

    if len(images) > 0:
        image_scroe_map = aes.inference(images)
        images = []

        for img_path, score in image_scroe_map.items():
            print(f"{img_path}: {score}")

            if score > limitscore:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                shutil.copy(img_path, output_dir)

                
