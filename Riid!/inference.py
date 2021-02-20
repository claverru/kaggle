import os
import math
import time
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import *
# from model_2 import *

from local_testing import test_generator


windows_size = 128
d_model = 256
num_heads = 4
n_encoder_layers = 3
n_decoder_layers = 3
bs = 64

df = pd.read_csv(
    'data/train.csv.zip',
    usecols=dtype.keys(),
    dtype=dtype,
    nrows=10**5
)
# df = df[df.answered_correctly!=-1]
df = df.groupby('user_id').tail(windows_size)

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

questions, part_ids, lecture_map, tag_ids = transform_questions(questions, lectures)
df, content_ids, task_container_ids = transform_df(df, questions, lecture_map)
df = {uid: u.drop(columns='user_id') for uid, u in df.groupby('user_id')}


s_infer = Inference(df, windows_size)
columns = list(s_infer.users_d[115].columns)
print(f'\n{columns}\n')

n_features = len(columns)

tf.keras.backend.clear_session()
model = tf.saved_model.load('models/model_21')


columns[columns.index('answered_correctly')] = 'user_id'
columns = [c for c in columns if c not in questions.columns] + ['row_id']


while True:
    iter_test = test_generator('data/tests')
    s = time.time()
    for test in iter_test:

        try:
            prior_correct = eval(test['prior_group_answers_correct'].iloc[0])
            # prior_correct = [a for a in prior_correct]# if a != -1]
        except:
            prior_correct = []

        if prior_correct:
            prior_test.insert(
                s_infer.c_indices['answered_correctly'], 
                'answered_correctly', 
                prior_correct
            )
            for uid, user in prior_test.groupby('user_id'):
                s_infer.update_user(uid, user.drop(columns='user_id'))

        # Save for later
        non_lectures_mask = test['content_type_id'] == 0
        
        # Filter test
        test = test[columns]

        # Add global features
        test, _, _ = transform_df(test, questions, lecture_map, ~non_lectures_mask)

        # Save test for later
        prior_test = test.drop(columns='row_id').copy()

        # Make x
        x = np.apply_along_axis(
            s_infer.get_user_for_inference,
            1,
            test.drop(columns='row_id').to_numpy()
        )

        # Predict
        predictions = []
        for i in range(math.ceil(len(x)/bs)):
            b = tf.convert_to_tensor(x[i*bs:(i+1)*bs], dtype=tf.float32)
            predictions.append(model.signatures['serving_default'](b)['output'])
        predictions = tf.concat(predictions, axis=0)

        # predictions = predictions[:, :, 1]
        # idx = np.any(x!=0, -1).sum(-1, keepdims=True)-1
        # predictions = np.take_along_axis(predictions.numpy(), idx, -1)

        predictions = predictions[:, -1, 1]
        
        test['answered_correctly'] = predictions
        
        # env.predict(test[['row_id', 'answered_correctly']])

    print(time.time()-s)