import os
import pickle
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from official.nlp.transformer.optimizer import LearningRateSchedule

from utils import *
from model import *


tf.get_logger().setLevel(logging.ERROR)

print('Loading DF')
df = pd.read_csv(
    'data/train.csv.zip',
    usecols=dtype.keys(),
    dtype=dtype,
    # nrows=10**6
)

df = df.set_index('user_id')

questions = pd.read_csv(
    'data/questions.csv', 
    dtype=dtype_questions,
    usecols=dtype_questions.keys(),
    index_col='question_id'
)

lectures = pd.read_csv(
    'data/lectures.csv', 
    dtype=dtype_lectures,
    usecols=dtype_lectures.keys(),
)

print('Transforming')
questions, part_ids, lecture_map, tag_ids = transform_questions(
    questions, lectures)
df, content_ids, task_container_ids = transform_df(df, questions, lecture_map)


print(df.dtypes)

windows_size = 176
d_model = 440
num_heads = 4
n_encoder_layers = 4
n_decoder_layers = 4

warm_steps = 6000
train_ratio = 0.98
epochs = 100
patience = 3
batch_size = 128
validation_freq = [18, 20, 22] + list(range(24, epochs+1))
s_train = RiidSequence(df, windows_size, batch_size, train_ratio)
s_val = RiidSequence(df, windows_size, batch_size, train_ratio, subset='val')

print(s_train.user_example().columns)
print(f'#users: {len(df.index.unique())}')
print(f'Batches in train: {len(s_train)}')
print(f'Batches in val: {len(s_val)}')

n_features = s_train[0][0].shape[-1]

learning_rate = LearningRateSchedule(0.1, d_model, warm_steps)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.95, beta_2=0.999, epsilon=1e-8, clipvalue=2.)

tf.keras.backend.clear_session()
model = get_series_model(
        n_features,
        content_ids,
        task_container_ids,
        part_ids,
        tag_ids,
        windows_size=windows_size,
        d_model=d_model,
        num_heads=num_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers
    )

model.compile(
    optimizer=optimizer, 
    loss={'a': loss_function_a, 'c': loss_function_c}, 
    metrics={'c': AUC()}
)

history = model.fit(
    s_train,
    validation_data=s_val,
    epochs=epochs,
    workers=4,
    shuffle=True,
    use_multiprocessing=True,
    validation_freq=validation_freq,
    callbacks=tf.keras.callbacks.EarlyStopping(
        patience=patience, 
        monitor='val_c_auc', 
        mode='max', 
        restore_best_weights=True),
)

model.save('model')
pickle.dump(
    history.history, open('history.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)