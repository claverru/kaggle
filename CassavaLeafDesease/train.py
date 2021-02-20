import os
import glob
import math

import timm
import torch
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from timm.data import create_transform
from sklearn.model_selection import RepeatedStratifiedKFold

from bitempered import bi_tempered_logistic_loss


class CassavaDataset(torch.utils.data.Dataset):

    def __init__(self, df, transforms=None):
        super().__init__()
        self.paths = df['image_path'].to_numpy()
        self.labels = pd.get_dummies(df['label']).to_numpy()
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        label = self.labels[index]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(img)
        return image, label
    

class CasavaDataModule(pl.LightningDataModule):

    def __init__(self, train_df, val_df, batch_size, img_size):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_df = train_df
        self.val_df = val_df

    def setup(self, stage=None):

        train_T = create_transform(
            input_size=self.img_size,
            is_training=True,
            color_jitter=0.25,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            vflip=0.5
        )
        val_T = create_transform(
            input_size=self.img_size,
            is_training=False,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        self.train_ds = CassavaDataset(self.train_df, transforms=train_T)
        self.val_ds = CassavaDataset(self.val_df, transforms=val_T)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=5,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=5
        )


class CasavaModel(pl.LightningModule):

    def __init__(self, 
                 arch, 
                 pretrained,
                 img_size,
                 n_classes=5,
                 lr=3e-4, 
                 plat_fac=0.5, 
                 plat_pat=1):
        super().__init__()

        self.model = timm.create_model(arch, pretrained=True)

        if 'deit' in arch or 'vit' in arch:

            old_stride_size = self.model.patch_embed.proj.stride[0]
            old_img_size = self.model.patch_embed.img_size[0]
            features = old_img_size**2//old_stride_size**2

            stride_size = math.ceil(math.sqrt(img_size**2//features))
            stride = (stride_size, stride_size)

            new_weight = torch.nn.functional.interpolate(
                self.model.patch_embed.proj.weight, 
                size=stride, 
                mode='bicubic', 
                align_corners=True
            )
            self.model.patch_embed.proj.weight = torch.nn.Parameter(new_weight)
            self.model.patch_embed.proj.stride = stride
            self.model.patch_embed.proj.kernel_size = stride
            self.model.patch_embed.img_size = (img_size, img_size)
            self.model.head = torch.nn.Linear(
                in_features=self.model.head.in_features, 
                out_features=n_classes, 
                bias=True
            )
        else:
            self.model.eval()
            try:
                self.model.fc = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(
                        in_features=self.model.fc.in_features, 
                        out_features=n_classes)
                )
            except:
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(
                        in_features=self.model.classifier.in_features, 
                        out_features=n_classes)
                )
        
        self.acc_object = pl.metrics.Accuracy()
        
        self.lr = lr
        self.plat_fac = plat_fac
        self.plat_pat = plat_pat

        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = bi_tempered_logistic_loss(y_pred, y_true, 0.7, 1.3, 0.1).mean()
        acc = self.acc_object(y_pred, y_true.argmax(-1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = bi_tempered_logistic_loss(y_pred, y_true, 0.7, 1.3, 0.1).mean()
        acc = self.acc_object(y_pred, y_true.argmax(-1))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.plat_fac, patience=self.plat_pat),
            'monitor': 'val_acc'
        }


ARCH = 'swsl_resnext50_32x4d'
# FOLDS = [4, 5, 6, 7, 8, 9]
FOLDS = list(range(10))
IMG_SIZE = 512
BATCH_SIZE = 32
LEARNING_RATE = 1.5e-4

TRAIN_IMG_DIR = 'data/train_images/'
OLD_TRAIN_IMG_DIR = 'data/olddata/train/**/**'


if __name__ == '__main__':

    df = pd.read_csv('data/train.csv')
    df['image_path'] = TRAIN_IMG_DIR + df.image_id

    suspicious = pd.read_csv('data/suspicious.csv')
    df['suspicious'] = suspicious['suspicious'].astype('uint8')+1
    df['strats'] = (df.label+1)*df.suspicious

    # Old data competition
    old_df = pd.DataFrame(glob.glob(OLD_TRAIN_IMG_DIR), columns=['image_path'])
    d = {'cbb': 0, 'cbsd': 1, 'cgm': 2, 'cmd': 3, 'healthy': 4}
    old_df['label'] = old_df.image_path.apply(lambda x: d[x.split(os.sep)[-2]])

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=152)

    for i, (train_idx, val_idx) in enumerate(skf.split(df, df['strats'])):

        if i not in FOLDS:
            continue

        print(f'Training Fold {i}')
        
        torch.cuda.empty_cache()
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Old data competition
        train_df = pd.concat([train_df, old_df])


        casava_data_module = CasavaDataModule(
            train_df, val_df, BATCH_SIZE, IMG_SIZE)
        
        casava_model = CasavaModel(ARCH, True, IMG_SIZE, 5, lr=LEARNING_RATE)
        
        trainer = pl.Trainer(
            accumulate_grad_batches=32,
            gpus=1, 
            benchmark=True,
            logger=pl.loggers.CSVLogger(
                save_dir='lightning_logs', 
                name=ARCH),
            precision=16,
            callbacks=[
                pl.callbacks.ProgressBar(),
                pl.callbacks.EarlyStopping(
                    monitor='val_acc', patience=5, mode='max'),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_acc',
                    mode='max', 
                    filename='{epoch}_{val_acc:.4f}')
            ]
        )
        
        trainer.fit(casava_model, casava_data_module)