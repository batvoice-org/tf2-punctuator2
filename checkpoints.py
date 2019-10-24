import os

import tensorflow as tf


def get_checkpoint(checkpoint_dir,
                      optimizer,
                      encoder,
                      decoder):
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    return checkpoint, checkpoint_prefix


def restore_checkpoint(checkpoint_dir, checkpoint):
    print("Restoring checkpoints.")
    latest_ckpt_file = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_ckpt_file)
