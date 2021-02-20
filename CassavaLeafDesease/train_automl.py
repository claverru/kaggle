import glob

import numpy as np
import torch
import pandas as pd
from flaml import AutoML

cs = ['0', '1', '2', '3', '4']

paths = sorted(glob.glob('preds/**/**'))

print('Loading DFs')
dfs = []
fold = []
for i, path in enumerate(paths):
    print(path)
    df = pd.read_csv(path)
    df[cs] = torch.softmax(torch.tensor(df[cs].to_numpy()), -1).numpy()
    fold.append(df)
    if (i+1)%5==0:
        df = pd.concat(fold).sort_values('image_id').reset_index(drop=True)
        if dfs:
            df = df.drop(columns=['image_id', 'label'])
        dfs.append(df)
        fold = []

df = pd.concat(dfs, axis=1)

print(df.head())

X = df.drop(columns=['label', 'image_id'])
y = df['label']


print(X.head())
print(X.shape)

print('Training')
automl = AutoML()

automl.fit(
    X.to_numpy(), 
    y.to_numpy(),
    time_budget=30000,
    max_iter=100,
    estimator_list=['lgbm'],
    metric='accuracy',
    task='classification',
    eval_method='cv',
    n_splits=5,
    # ensemble=True
)


import pickle

pickle.dump(automl.model, open('L2.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)