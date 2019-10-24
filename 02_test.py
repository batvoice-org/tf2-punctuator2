#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K

from pipeline import *
from model import *
from optimization import *
from checkpoints import *
from metrics import *

try:
    print("Using device: ", os.environ["CUDA_VISIBLE_DEVICES"])
except KeyError:
    pass

ALL_DATA = os.environ["ALL_DATA"]
TEST_DATA = os.environ["TEST_DATA"]
ENCODER_PATH_PREFIX = os.environ["ENCODER_PATH_PREFIX"]
CHECKPOINTS_PATH = os.environ["CHECKPOINTS_PATH"]
SAVED_MODEL_PATH = os.environ["SAVED_MODEL_PATH"]
EMBEDDING_DIM = int(os.environ["EMBEDDING_DIM"])
N_UNITS = int(os.environ["N_UNITS"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
INIT_LEARNING_RATE = float(os.environ["INIT_LEARNING_RATE"])
TRAINING = bool(os.environ["TRAINING"])

exp_dir = os.environ["EXP_DIR"]

ckpt_dir = os.path.join(exp_dir, "training_checkpoints")

TARGET_VOCAB_SIZE = 1

input_encoder = tfds.features.text.TokenTextEncoder.load_from_file(ENCODER_PATH_PREFIX)
pipeline = Pipeline(TEST_DATA, ENCODER_PATH_PREFIX, encoder=input_encoder)

MAX_SEQ_LEN, NUM_ELEMENTS = pipeline.get_max_seq_len_and_num_elems()
    
test_ds, _ = pipeline.get_dataset(padded_shapes=([MAX_SEQ_LEN], [MAX_SEQ_LEN]))
print("Max sequence length in test (num words): ", MAX_SEQ_LEN)
print("Number of test examples: ", NUM_ELEMENTS)


encoder = BidirectionalEncoder(input_encoder.vocab_size, 
                  EMBEDDING_DIM, N_UNITS, BATCH_SIZE)

decoder = PunctDecoder(TARGET_VOCAB_SIZE, N_UNITS, BATCH_SIZE)

print("Initial learning rate: ", INIT_LEARNING_RATE)
optimizer, loss_function = get_optimizer_and_loss(INIT_LEARNING_RATE,)

checkpoint_dir = os.path.join(exp_dir, './training_checkpoints')
checkpoint, checkpoint_prefix = get_checkpoint(checkpoint_dir, optimizer,
                                                  encoder, decoder)

def test_step_py(inp, targ, enc_init_state, dec_init_state, precision_object, recall_object):
  loss = 0

  precision_object.reset_states()
  recall_object.reset_states()

  print("Tracing computation graph.")

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

  return batch_loss, batch_recall, batch_precision


test_step = tf.function(func=test_step_py,)

print("Beginning testing.")

restore_checkpoint(checkpoint_dir, checkpoint)

STEPS = NUM_ELEMENTS // BATCH_SIZE

print("STEPS: ", STEPS)

start = time.time()

enc_init_state = encoder.initialize_hidden_state()
dec_init_state = decoder.initialize_hidden_state()

thresholds = list(np.arange(.1, 1., .1))

precision_object = tf.keras.metrics.Precision(thresholds=thresholds)
recall_object = tf.keras.metrics.Recall(thresholds=thresholds)

total_loss = 0
total_recall = 0
total_precision = 0

for (batch, (inp, targ)) in enumerate(test_ds.take(STEPS)):

   batch_loss, batch_recall, batch_precision = test_step(inp, targ, enc_init_state, dec_init_state,
                                                       precision_object, recall_object)

   total_loss += batch_loss

   total_recall += batch_recall
   total_precision += batch_precision

   if batch % 50 == 0:
       print('Batch {} Loss {:.4f} Recall w/ threshold .5 {:.4f} Precision w/ threshold .5 {:.4f} '.format(
                                                 batch,
                                                 batch_loss.numpy(),
                                                 batch_recall[4].numpy(),
                                                 batch_precision[4].numpy()))


recall = total_recall / STEPS
precision = total_precision / STEPS
print('Test loss: {:.4f}'.format(total_loss / STEPS))

for thresh, rec, prec in zip(thresholds, list(recall.numpy()), list(precision.numpy())):
    print('Classification threshold {:.2f} Recall: {:.4f} Precision: {:.4f} F1: {:4f}'.format(
        thresh, rec, prec, f1(prec, rec)
    ))

print("Done testing.")
