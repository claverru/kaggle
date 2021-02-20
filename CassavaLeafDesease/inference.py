import os
import glob
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from train_torch import CasavaDataModule, TRAIN_IMG_DIR, BATCH_SIZE, IMG_SIZE


arch_seed = {
    'resnest101e': 43,
    'swsl_resnext101_32x4d': 28,
    'swsl_resnext50_32x4d': 152

}

df = pd.read_csv('data/train.csv')
df['image_path'] = TRAIN_IMG_DIR + df.image_id

suspicious = pd.read_csv('data/suspicious.csv')
df['suspicious'] = suspicious['suspicious'].astype('uint8')+1
df['strats'] = (df.label+1)*df.suspicious

for arch, seed in arch_seed.items():
    if '50' not in arch:
        continue

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    
    for i, (train_idx, val_idx) in enumerate(skf.split(df, df['strats'])):
        path = f'models/{arch}/{i}.pt'
        print(f'Inference: {path}')

        
        train_df = df.iloc[train_idx].copy()
        
        val_df = df.iloc[val_idx].copy()

        casava_data_module = CasavaDataModule(
            train_df, val_df, BATCH_SIZE, IMG_SIZE)
        
        casava_data_module.setup()
        
        val_dataloader = casava_data_module.val_dataloader()
        
        torch.cuda.empty_cache()
        model = torch.jit.load(path)
        model = model.cuda()
        
        y_pred = []
        y_true = []

        for x, y in tqdm(val_dataloader, ncols=100):
            pred = model(x.to('cuda')).data.cpu()
            y_pred.append(pred)
            y_true.append(y)

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        print('Acc: ', (y_pred.argmax(-1)==y_true.argmax(-1)).mean())

        preds_df = pd.DataFrame(y_pred, columns=list(range(5)))
        preds_df['image_id'] = val_df['image_id'].to_numpy()
        preds_df['label'] = val_df['label'].to_numpy()

        pred_dir = f'preds/{arch}'
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        pred_path = os.path.join(pred_dir, f'{i}.csv')

        preds_df.to_csv(pred_path, index=False)