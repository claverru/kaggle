import os
import glob
from pathlib import Path

import timm
import torch
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import RepeatedStratifiedKFold

from train_torch import CasavaDataModule, TRAIN_IMG_DIR, OLD_TRAIN_IMG_DIR


class ResNext101(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = timm.create_model('swsl_resnext101_32x4d', pretrained=False)
        self.model.eval()
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=2048, out_features=5)
        )      
        

    def forward(self, x):
        x = self.model(x)
        return x
    

class ResNest101(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnest101e', pretrained=False)
        self.model.eval()
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=2048, out_features=5)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ViT(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_384', pretrained=False)
        new_weight = torch.nn.functional.interpolate(
            self.model.patch_embed.proj.weight, 
            size=(20, 20), 
            mode='bicubic', 
            align_corners=True
        )
        self.model.patch_embed.proj.weight = torch.nn.Parameter(new_weight)
        self.model.patch_embed.proj.stride = (20, 20)
        self.model.patch_embed.proj.kernel_size = (20, 20)
        self.model.patch_embed.img_size = (480, 480)
        self.model.head = torch.nn.Linear(
            in_features=self.model.head.in_features, out_features=5)

    def forward(self, x):
        x = self.model(x)
        return x



class ResNext50(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = timm.create_model('swsl_resnext50_32x4d', pretrained=False)
        self.model.eval()
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=2048, out_features=5)
        )      
        
    def forward(self, x):
        x = self.model(x)
        return x


def load_model(path):
    print('Loading', path)
    if 'resnest101' in path:
        return ResNest101.load_from_checkpoint(path)
    elif 'resnext101' in path:
        return ResNext101.load_from_checkpoint(path)
    elif 'vit' in path:
        return ViT.load_from_checkpoint(path)
    elif '50' in path:
        return ResNext50.load_from_checkpoint(path)
    else:
        return 'NO'


if __name__ == '__main__':

    paths = sorted(glob.glob('lightning_logs/**/**/checkpoints/**'))

    for path in paths:

        if 'resnext50' not in path:
            continue

        torch.cuda.empty_cache()
        model = load_model(path)
        model = model.cpu()
        model.freeze()
        arch = path.split(os.sep)[1]
        fold = path.split(os.sep)[2].split('_')[-1]
        new_dir = f'models/{arch}'
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        new_path = os.path.join(new_dir, f'{fold}.pt')

        script = model.to_torchscript()
        torch.jit.save(script, new_path)
