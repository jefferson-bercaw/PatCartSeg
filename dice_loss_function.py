import tensorflow as tf
from get_data_path import get_data_path
import os
from PIL import Image
import numpy as np


def dice_coefficient(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def weighted_dice_loss(y_true, y_pred):
    # Weight given to pixels on articulating surface of patella and cartilage
    n = 10

    # Convert to p, pc, psurf, pcsurf predictions and labels (2d tensor)
    y_true = tf.reshape(y_true, (-1, 4))
    y_pred = tf.reshape(y_pred, (-1, 4))

    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    union = tf.reduce_sum(y_true, axis=0) + tf.reduce_sum(y_pred, axis=0)
    dice_loss = 1 - (2 * intersection + 1) / (union + 1)

    weights = tf.constant([1, 1, n, n], dtype=tf.float32)
    weighted_dice_loss = dice_loss * weights
    return tf.reduce_sum(weighted_dice_loss)


if __name__ == "__main__":
    results_path = "R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\results\\unet_2024-07-03_16-16-58_cHT5"
    img = "AS_006-0053.bmp"

    p_pred_loc = os.path.join(results_path, "pat", img)
    pc_pred_loc = os.path.join(results_path, "pat_cart", img)

    pat_pred = np.array(Image.open(p_pred_loc))
    pat_cart_pred = np.array(Image.open(pc_pred_loc))
    y_pred = np.zeros((256, 256, 2))
    y_pred[:, :, 0] = np.float64(pat_pred)
    y_pred[:, :, 1] = np.float64(pat_cart_pred)

    # Load in ground truth
    label_path = get_data_path("cHT")
    image_path = os.path.join(label_path, "test", "mask", img)
    label_image = np.array(Image.open(image_path))

    y_true = np.zeros((256, 256, 2))
    y_true[:, :, 0] = label_image == 1
    y_true[:, :, 1] = label_image == 2

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)

    loss = weighted_dice_loss(y_true, y_pred)
    print(loss)