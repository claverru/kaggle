import tensorflow as tf

from official.nlp.keras_nlp.layers import (
  TransformerEncoderBlock,
  PositionEmbedding
)
from official.nlp.modeling.layers import TransformerDecoderBlock


def create_padding_mask(seqs):
  mask = tf.cast(tf.reduce_all(tf.math.equal(seqs, 0.), axis=-1), tf.float32)
  return mask[:, tf.newaxis, :]


def create_look_ahead_mask(size, shift_right=0):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, shift_right)
  return mask


def create_bundle_mask(task, windows_size):
  es = '...b, ...c -> ...bc'
  return tf.cast(
    tf.einsum(es, task, task) != tf.square(task)[:, tf.newaxis, :], 
    tf.float32
  )+tf.eye(windows_size)

def get_series_model(
        n_features,
        content_ids,
        task_container_ids,
        part_ids,
        tag_ids,
        windows_size=64,
        d_model=24,
        num_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2
    ):
    # Input
    inputs = tf.keras.Input(shape=(windows_size, n_features), name='inputs')    
    
    # Divide branches
    lag, content, content_type, task, user_ans, \
    ans_correctly, lapse, exp, correct_ans, part = \
      tf.unstack(inputs[..., :-6], axis=-1)
    tags = inputs[..., -6:]

    # Masks
    pad_mask = create_padding_mask(inputs)
    la_mask = create_look_ahead_mask(windows_size)
    bundle_mask = create_bundle_mask(task, windows_size)
    mask = (1-tf.maximum(pad_mask, la_mask))*bundle_mask
    
    # Create embeddings
    answer_emb_layer = tf.keras.layers.Embedding(7, d_model)
    user_ans_emb = answer_emb_layer(user_ans)
    correct_ans_emb = answer_emb_layer(correct_ans)

    content_type_emb = tf.keras.layers.Embedding(3, d_model)(content_type)
    content_emb = tf.keras.layers.Embedding(content_ids, d_model)(content)
    # task_emb = tf.keras.layers.Embedding(task_container_ids, d_model)(task)
    task_emb = tf.keras.layers.experimental.EinsumDense(
      '...x,xy->...y', d_model)(task[..., tf.newaxis])
    
    ans_correctly_emb = tf.keras.layers.Embedding(5, d_model)(ans_correctly)
    exp_emb = tf.keras.layers.Embedding(4, d_model)(exp)
    part_emb = tf.keras.layers.Embedding(part_ids, d_model)(part)
    
    tags = tf.ragged.boolean_mask(tags, tags!=0)
    tags_emb = tf.keras.layers.Embedding(tag_ids, d_model)(tags)
    tags_emb = tags_emb.to_tensor(shape=(None, windows_size, None, d_model))
    tags_emb = tf.reduce_sum(tags_emb, -2)

    # Time features
    time = tf.stack([lag, lapse], -1)
    time_emb = tf.keras.layers.experimental.EinsumDense(
      '...x,xy->...y', d_model)(time)
   
   # Position 
    pos_emb = PositionEmbedding(windows_size)(content_emb)

    # Add emb
    e = tf.keras.layers.Add()([
      pos_emb,
      content_emb,
      correct_ans_emb,
      content_type_emb,
      part_emb,
      tags_emb
    ])

    d = tf.keras.layers.Add()([
      pos_emb,
      ans_correctly_emb,
      user_ans_emb,
      exp_emb,
      task_emb,
      time_emb
    ])

    for _ in range(n_encoder_layers):
      e = TransformerEncoderBlock(
          num_heads,
          d_model*2,
          'swish',
          output_dropout=0.2,
          attention_dropout=0.1,
          inner_dropout=0.4
      )([e, mask])

    for _ in range(n_encoder_layers):
      d, _ = TransformerDecoderBlock(
          num_heads,
          d_model*2,
          'swish',
          dropout_rate=0.2,
          attention_dropout_rate=0.1,
          intermediate_dropout=0.4
      )([d, e, mask, mask])


    output_a = tf.keras.layers.Dense(
      4, activation='softmax', name='a')(d)

    correct_ids = tf.cast(tf.where(correct_ans>1, correct_ans-2, 0), tf.int32)

    output_c = tf.gather(output_a, correct_ids, batch_dims=2)
    output_c = tf.keras.layers.Lambda(tf.identity, name='c')(output_c)
    
    return tf.keras.Model(inputs, [output_a, output_c], name='model')


loss_object_a = tf.keras.losses.SparseCategoricalCrossentropy(
  reduction=tf.keras.losses.Reduction.NONE)

def loss_function_a(real, pred):
  mask = tf.not_equal(real, 4)

  real = tf.clip_by_value(real, 0, 3)
  pred = tf.where(tf.math.is_finite(pred), pred, 0.5)

  loss_ = loss_object_a(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


"""
loss_object_c = tf.keras.losses.SparseCategoricalCrossentropy(
  reduction=tf.keras.losses.Reduction.NONE)

def loss_function_c(real, pred):
  mask = tf.not_equal(real, 2)

  real = tf.clip_by_value(real, 0, 1)
  pred = tf.where(tf.math.is_finite(pred), pred, 0.5)

  loss_ = loss_object_c(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
"""


loss_object_c = tf.keras.losses.BinaryCrossentropy(
  reduction=tf.keras.losses.Reduction.NONE)

def loss_function_c(real, pred):
  mask = tf.not_equal(real, 2)

  real = tf.clip_by_value(real, 0, 1)
  pred = tf.where(tf.math.is_finite(pred), pred, 0)

  real = real[..., tf.newaxis]
  pred = pred[..., tf.newaxis]

  loss_ = loss_object_c(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


class AUC(tf.keras.metrics.AUC):

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = y_true[:, -1]
    y_pred = y_pred[:, -1]#, 1]
    # y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, 0)
    return super(AUC, self).update_state(
      y_true, y_pred, sample_weight=sample_weight)