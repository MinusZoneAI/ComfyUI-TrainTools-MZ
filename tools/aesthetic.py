import json
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

from warnings import filterwarnings

import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image


from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class Aestheic:
    def __init__(self, clip_path, aes_model):

        model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        # load the model you trained previously or the model available in this repo
        s = torch.load(aes_model)

        model.load_state_dict(s)

        model.to("cuda")
        model.eval()

        self.model = model

        self.device = "cuda"

        model2 = CLIPModel.from_pretrained(
            clip_path, local_files_only=True).to("cuda")
        preprocess = CLIPProcessor.from_pretrained(
            clip_path, local_files_only=True)

        self.model2 = model2
        self.preprocess = preprocess

    def prediction(self, img_paths):
        if not isinstance(img_paths, list):
            img_paths = [img_paths]

        pil_images = []
        for img_path in img_paths:
            pil_image = Image.open(img_path)
            pil_images.append(pil_image)

        inputs = self.preprocess(
            images=pil_images, return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            image_features = self.model2.get_image_features(**inputs)
            print(image_features.shape)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())

        prediction = self.model(torch.from_numpy(
            im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))

        prediction_score = prediction.cpu().detach().numpy()
        prediction_score = prediction_score.flatten()

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
                        default="D:\\数据集\\clip-vit-large-patch14")
    parser.add_argument("-m", "--aes_model", type=str,
                        default="D:\\数据集\\sac+logos+ava1-l14-linearMSE.pth")
    parser.add_argument("-s", "--score", type=float, default=5.9)

    parser.add_argument("-b", "--batch_size", type=int, default=1)

    args = parser.parse_args()
    limitscore = args.score

    aes = Aestheic(args.clip_path, args.aes_model)

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

                if len(images) < batch_size:
                    images.append(file_path)
                else:
                    image_scroe_map = aes.prediction(images)
                    images = []
                    # print(scroe)
                    # print(json.dumps(scroe, indent=4, ensure_ascii=False))
                    # exit(0)

                    for img_path, score in image_scroe_map.items():
                        print(f"{img_path}: {score}")

                        if score > limitscore:
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            shutil.copy(img_path, output_dir)

    if len(images) > 0:
        aes.prediction(images)
        images = []

        for img_path, score in image_scroe_map.items():
            print(f"{img_path}: {score}")

            if score > limitscore:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                shutil.copy(img_path, output_dir)

                
