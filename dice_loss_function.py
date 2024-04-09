import tensorflow as tf


def dice_coefficient(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
