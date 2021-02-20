import random
import collections

import numpy as np
import pandas as pd
import tensorflow as tf


dtype = {
    'answered_correctly': 'int8',
    # 'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}

dtype_questions = {
    'question_id': 'int32',
    # 'bundle_id': 'int32',
    'correct_answer': 'int8',
    'part': 'int8',
    'tags': 'object',
}

dtype_lectures = {
    'lecture_id': 'int32',
    'tag': 'int16',
    'part': 'int8',
    # 'type_of': 'object'
}


def transform_questions(questions, lectures):
  # tags
  tag_names = [f'tag{i}' for i in range(6)]
  questions[tag_names] = tf.keras.preprocessing.sequence.pad_sequences(
    questions.tags.fillna('').str.split().to_numpy(), 
    dtype=np.int16, 
    value=-1
  )+1
  questions.drop(columns='tags', inplace=True)
  
  lectures[tag_names] = tf.keras.preprocessing.sequence.pad_sequences(
    lectures.tag.apply(lambda x: [x]).to_numpy(), 
    dtype=np.int16,
    maxlen=6,
    value=-1
  )+1
  lectures.drop(columns='tag', inplace=True)
  lectures['correct_answer'] = -1
  lectures['correct_answer'] = lectures['correct_answer'].astype('int8')

  # Concat both
  start = len(questions)
  lecture_map = \
    dict(zip(lectures.lecture_id, range(start, len(lectures)+start)))
  lectures.drop(columns='lecture_id', inplace=True)
  lectures.index = lecture_map.values()
  questions = pd.concat([questions, lectures])
  questions[tag_names] = questions[tag_names].astype('uint8')
  
  # Embeddings
  part_ids = questions.part.max()+1
  tag_ids = questions[tag_names].max().max()+1
  return questions, part_ids, lecture_map, tag_ids


def transform_df(df, questions, lecture_map, lectures_mask=None):

  if lectures_mask is None:
    lectures_mask = df.answered_correctly==-1

  df.loc[lectures_mask, 'content_id'] = \
    df.loc[lectures_mask, 'content_id'].map(lecture_map)

  df['prior_question_had_explanation'] = (
    df.prior_question_had_explanation.astype('float32').fillna(2)+1
  ).astype('int8')

  df['timestamp'] = df['timestamp'].astype('float32')
  df['prior_question_elapsed_time'] = \
    df['prior_question_elapsed_time'].astype('float32')

  df['prior_question_elapsed_time'] = np.log1p(df.prior_question_elapsed_time)

  df['prior_question_elapsed_time'] = \
    df.prior_question_elapsed_time.fillna(2.714793e+00)
  df['prior_question_elapsed_time'] = \
    (df.prior_question_elapsed_time-2.714793e+00)/4.546208e+00

  content_ids = questions.index.max()+2
  df = df.join(questions, on='content_id')
  df['content_type_id'] += 1
  df['content_id'] = df['content_id'].astype('int16')
  df['content_id'] += 1
  # df['task_container_id'] += 1
  df['task_container_id'] /= 10000
  task_container_ids = 10001
  return df, content_ids, task_container_ids


def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

def transform_ts(np_ts):
    log_delta = np.concatenate([[3.656529e+00], np.log1p(np.diff(np_ts))])
    log_delta = fill_zeros_with_last(log_delta)
    u, c = np.unique(np_ts, return_counts=True)
    div = np.where(np_ts == u[:, np.newaxis], c[:, np.newaxis], 0).sum(0)
    return (log_delta-3.656529e+00)/5.163177e+00/div


def add_features_to_user(user):
  user['timestamp'] = transform_ts(user.timestamp.to_numpy())
  user['answered_correctly'] = user['answered_correctly'].shift(fill_value=2)+2
  user['user_answer']  = user['user_answer'].shift(fill_value=4)
  user[['user_answer', 'correct_answer']] += 2
  return user


class RiidSequence(tf.keras.utils.Sequence):

  def __init__(self, 
               df, 
               windows_size,
               batch_size=512,
               train_ratio=0.8,
               subset='train'):
    self.df = df
    self.windows_size = windows_size
    self.batch_size = batch_size
    self.subset = subset
    
    user_indices = self.df.index.unique()
    idx = int(len(user_indices)*train_ratio)
    
    if self.subset == 'train':
      self.user_indices = user_indices[:idx]
      c = self.df.index.value_counts().clip(upper=1000)
      c = c[c.index.isin(self.user_indices)]
      self.c = c/c.sum()
    else:
      self.user_indices = user_indices[idx:]
    
    self.mapper = self._build_mapper()
  
  def _build_mapper(self):

    if self.subset == 'train':
      indices = np.random.choice(
        self.c.index, len(self.user_indices), p=self.c).tolist()
      a, b = np.unique(indices, return_counts=True)
      ids = np.random.choice(len(a), len(a), replace=False)
      a, b = a[ids], b[ids]
      indices = np.repeat(a, b).tolist()
    else:
      indices = []
      for uid in self.user_indices:
        indices.extend([(uid, idx) for idx in range(len(self.df.loc[[uid]]))])
    
    li = len(indices)
    return [indices[i:i+self.batch_size] for i in range(0, li, self.batch_size)]

  def on_epoch_end(self):
    self.mapper = self._build_mapper()

  def __len__(self):
    return len(self.mapper)

  def __getitem__(self, idx):
    if self.subset == 'train':
      x, ya, yc = self.get_train(idx)
    else:
      x, ya, yc = self.get_val(idx)
    x = tf.keras.preprocessing.sequence.pad_sequences(
      x, self.windows_size, dtype='float32', padding='pre', value=0.)
    ya = tf.keras.preprocessing.sequence.pad_sequences(
      ya, self.windows_size, dtype='uint8', padding='pre', value=4)
    yc = tf.keras.preprocessing.sequence.pad_sequences(
      yc, self.windows_size, dtype='uint8', padding='pre', value=2)
    return x, [ya, yc]

  def user_example(self):
    user = self.df.loc[[115]].copy()
    return add_features_to_user(user)

  def get_x_y(self, uid):
    user = self.df.loc[[uid]].copy()
    
    uya = user['user_answer'].copy().to_numpy()
    uya[uya==-1] = 4
    
    uyc = user['answered_correctly'].copy().to_numpy()
    uyc[uyc==-1] = 2
    
    ux = add_features_to_user(user).to_numpy()
    return ux, uya, uyc

  def get_train(self, idx):
    x, ya, yc = [], [], []

    for uid, n in np.array(np.unique(self.mapper[idx], return_counts=True)).T:
      ux, uya, uyc = self.get_x_y(uid)
      limit = max(1, len(ux)-self.windows_size+1)
      idxs = np.random.choice(limit, n)
      
      for idx in idxs:
        x.append(ux[idx:idx+self.windows_size])
        ya.append(uya[idx:idx+self.windows_size])
        yc.append(uyc[idx:idx+self.windows_size])

    return x, ya, yc

  def get_val(self, idx):
    x, ya, yc = [], [], []

    mapper = {}
    for uid, idx in self.mapper[idx]:  
      mapper.setdefault(uid, []).append(idx)

    for uid in mapper:
      ux, uya, uyc = self.get_x_y(uid)
      for idx in mapper[uid]:
        if uyc[idx] == 2: # 2 means lecture
          continue
        high = idx+1
        low = max(0, high-self.windows_size)
        x.append(ux[low:high])
        ya.append(uya[low:high])
        yc.append(uyc[low:high])

    return x, ya, yc

class Inference():

  def __init__(self, users_d, windows_size):
    self.users_d = users_d
    self.windows_size = windows_size
    self.c_indices = {
      c: i for i, c in enumerate(self.users_d[115].columns)}

  def get_user_for_inference(self, user_row):
    ac = self.c_indices['answered_correctly']
    ua = self.c_indices['user_answer']
    t = self.c_indices['timestamp']
    ca = self.c_indices['correct_answer']
    
    uid = user_row[ac]
    user_row[ac] = 2

    user_row = user_row[np.newaxis, ...]

    if uid in self.users_d:
      x = np.concatenate([self.users_d[uid], user_row])
    else:
      x = user_row
    
    x[:, ac] = np.roll(x[:, ac], 1) + 2
    x[:, ua] = np.roll(x[:, ua], 1) + 2
    x[:, ca] += 2
    x[:, t] = transform_ts(x[:, t])
    
    if x.shape[0] < self.windows_size:
      return np.pad(x, [[self.windows_size-x.shape[0], 0], [0, 0]])
    elif x.shape[0] > self.windows_size:
      return x[-self.windows_size:]
    else:
      return x

  def update_user(self, uid, user):
    if uid in self.users_d:
      self.users_d[uid] = \
        np.concatenate([self.users_d[uid], user])[-self.windows_size-5:]
    else:
      self.users_d[uid] = user