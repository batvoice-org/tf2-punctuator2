#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
TARGET_VOCAB_SIZE = 1

exp_dir = os.environ["EXP_DIR"]
input_encoder = tfds.features.text.TokenTextEncoder.load_from_file(ENCODER_PATH_PREFIX)
tokenizer = tfds.features.text.Tokenizer(reserved_tokens=["<B>"])


encoder = BidirectionalEncoder(input_encoder.vocab_size, 
                  EMBEDDING_DIM, N_UNITS, BATCH_SIZE)

decoder = PunctDecoder(TARGET_VOCAB_SIZE, N_UNITS, BATCH_SIZE)

print("Initial learning rate: ", INIT_LEARNING_RATE)
optimizer, loss_function = get_optimizer_and_loss(INIT_LEARNING_RATE,)

checkpoint_dir = os.path.join(exp_dir, './training_checkpoints')
checkpoint, checkpoint_prefix = get_checkpoint(checkpoint_dir, optimizer,
                                                  encoder, decoder)

restore_checkpoint(checkpoint_dir, checkpoint)

ckpt_dir = os.path.join(exp_dir, "training_checkpoints")

TARGET_VOCAB_SIZE = 1

def plot(sentence: str, all_weights: list, preds: list):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(np.array(all_weights), cmap='viridis')
    fig.colorbar(cax)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence.split(" "), fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + [str(pred) for pred in preds], fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("att.png")

def infer(sentence: str, threshold=.5):
 
    enc_init_state = [tf.zeros((1, N_UNITS)), tf.zeros((1, N_UNITS))]
    dec_init_state = tf.zeros((1, N_UNITS))

    tokenized = tokenizer.tokenize(sentence)
    tokenized = " ".join(tokenized)
    inputs = input_encoder.encode(tokenized)
    inputs = tf.expand_dims(tf.convert_to_tensor(inputs), 0)

    enc_output, _ = encoder(inputs, enc_init_state)    

    s_t_m = dec_init_state
    outputs = []
    all_weights = []
    for t in range(0, enc_output.shape[1]):
       y_t, s_t, att_weights = decoder(enc_output[:,t,:], s_t_m, enc_output, dec_init_state, return_weights=True)
       s_t_m = s_t
       outputs.append(y_t)
       all_weights.append(att_weights[0,:,0].numpy())
    outputs = [o.numpy()[0][0] for o in outputs]
    preds = [int(o > threshold) for o in outputs]
    plot(sentence, all_weights, preds)

if __name__ == "__main__":
    infer("bonjour comment puisje vous aider oui bonjour")
