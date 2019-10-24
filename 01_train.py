#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K

from pipeline import *
from model import *
from optimization import *
from checkpoints import *

try:
    print("Using device: ", os.environ["CUDA_VISIBLE_DEVICES"])
except KeyError:
    pass

ALL_DATA = os.environ["ALL_DATA"]
TRAIN_DATA = os.environ["TRAIN_DATA"]
ENCODER_PATH_PREFIX = os.environ["ENCODER_PATH_PREFIX"]
CHECKPOINTS_PATH = os.environ["CHECKPOINTS_PATH"]
EMBEDDING_DIM = int(os.environ["EMBEDDING_DIM"])
N_UNITS = int(os.environ["N_UNITS"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
INIT_LEARNING_RATE = float(os.environ["INIT_LEARNING_RATE"])
TRAINING = bool(os.environ["TRAINING"])

restoring = False
exp_dir = os.environ["EXP_DIR"]
if TRAINING:
    print("Training mode.")
    try:
        os.makedirs(exp_dir)
    except:
        restoring = True

ckpt_dir = os.path.join(exp_dir, "training_checkpoints")

TARGET_VOCAB_SIZE = 1 # binary

if TRAINING:
    if restoring:
        input_encoder = tfds.features.text.TokenTextEncoder.load_from_file(ENCODER_PATH_PREFIX)
        pipeline = Pipeline(TRAIN_DATA, ENCODER_PATH_PREFIX, encoder=input_encoder)
    else:
        pipeline = Pipeline(TRAIN_DATA, ENCODER_PATH_PREFIX)

    MAX_SEQ_LEN, NUM_ELEMENTS = pipeline.get_max_seq_len_and_num_elems()
    
    train_ds, input_encoder = pipeline.get_dataset(padded_shapes=([MAX_SEQ_LEN], [MAX_SEQ_LEN]))
    print("Max sequence length in train (num words): ", MAX_SEQ_LEN)
    print("Number of training examples: ", NUM_ELEMENTS)


encoder = BidirectionalEncoder(input_encoder.vocab_size, 
                  EMBEDDING_DIM, N_UNITS, BATCH_SIZE)

decoder = PunctDecoder(TARGET_VOCAB_SIZE, N_UNITS, BATCH_SIZE)

print("Initial learning rate: ", INIT_LEARNING_RATE)

optimizer, loss_function = get_optimizer_and_loss(INIT_LEARNING_RATE,)


checkpoint_dir = os.path.join(exp_dir, './training_checkpoints')
checkpoint, checkpoint_prefix = get_checkpoint(checkpoint_dir, optimizer,
                                                  encoder, decoder)


def train_step_py(inp, targ, enc_init_state, dec_init_state, precision_object, recall_object):
  loss = 0

  precision_object.reset_states()
  recall_object.reset_states()

  print("Tracing computation graph.")

  with tf.GradientTape() as tape:
    enc_output, _ = encoder(inp, enc_init_state)

    s_t_m = dec_init_state
    
    for t in range(0, enc_output.shape[1]):
    
      y_t, s_t = decoder(enc_output[:,t,:], s_t_m, enc_output, dec_init_state)
      loss += loss_function(targ[:, t], y_t)
      precision_object.update_state(K.clip(targ[:, t], 0, 1), K.clip(tf.reshape(y_t, targ[:, t].shape), 0, 1))
      recall_object.update_state(K.clip(targ[:, t], 0, 1), K.clip(tf.reshape(y_t, targ[:, t].shape), 0, 1))

      s_t_m = s_t

  batch_loss = (loss / int(targ.shape[1]))
  batch_precision = precision_object.result()
  batch_recall = recall_object.result()

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss, batch_recall, batch_precision


train_step = tf.function(func=train_step_py,)

if TRAINING:
    print("Beginning training.")
    if restoring:
        restore_checkpoint(checkpoint_dir, checkpoint)

    EPOCHS = 100
    STEPS_PER_EPOCH = NUM_ELEMENTS // BATCH_SIZE

    print("STEPS_PER_EPOCH: ", STEPS_PER_EPOCH)

    for epoch in range(EPOCHS):
      start = time.time()

      enc_init_state = encoder.initialize_hidden_state()
      dec_init_state = decoder.initialize_hidden_state()
      precision_object = tf.keras.metrics.Precision(thresholds=.5)
      recall_object = tf.keras.metrics.Recall(thresholds=.5)

      total_loss = 0

      for (batch, (inp, targ)) in enumerate(train_ds.take(STEPS_PER_EPOCH)):

        batch_loss, batch_recall, batch_precision = train_step(inp, targ, enc_init_state, dec_init_state, 
                                                               precision_object, recall_object)
        total_loss += batch_loss

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Recall {:.4f} Precision {:.4f} '.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy(),
                                                         batch_recall.numpy(),
                                                         batch_precision.numpy()))

      checkpoint.save(file_prefix = checkpoint_prefix)

      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / STEPS_PER_EPOCH))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

print("Done training.")
