import tensorflow as tf


def get_optimizer_and_loss(init_learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_learning_rate)
    loss_object = tf.keras.losses.BinaryCrossentropy()

    def loss_function(real, pred):
        loss_ = loss_object(real, pred)
        return tf.reduce_mean(loss_)

    return optimizer, loss_function