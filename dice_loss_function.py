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

    # Identify furthest-right pixels
    p_true = y_true[:, :, 0]
    pc_true = y_true[:, :, 1]

    p_pred = y_pred[:, :, 0]
    pc_pred = y_pred[:, :, 1]

    # Change each true pixel to the column number it's in
    col_indices = tf.range(256, dtype=tf.float32)
    col_indices = tf.reshape(col_indices, (1, 256))
    col_indices = tf.tile(col_indices, (256, 1))

    p_col_locs = col_indices * p_true
    pc_col_locs = col_indices * pc_true

    right_p = tf.argmax(p_col_locs, axis=1)
    right_pc = tf.argmax(pc_col_locs, axis=1)

    p_row = tf.reshape(tf.where(right_p > 0), (-1))
    p_col = tf.gather(right_p, p_row)

    pc_row = tf.reshape(tf.where(right_pc > 0), (-1))
    pc_col = tf.gather(right_pc, pc_row)

    p_inds = tf.stack([p_row, p_col], axis=1)
    pc_inds = tf.stack([pc_row, pc_col], axis=1)

    # Values of the right-most true labels
    right_p_true = tf.gather_nd(p_true, p_inds)
    right_pc_true = tf.gather_nd(pc_true, pc_inds)

    # Corresponding values of the predicted label
    right_p_pred = tf.gather_nd(p_pred, p_inds)
    right_pc_pred = tf.gather_nd(pc_pred, pc_inds)

    # Calculate 3 dice loss scores, and weight accordingly
    # 1. Patella right coordinates
    p_right_loss = dice_loss(right_p_true, right_p_pred)

    # 2. Patellar cartilage right coordinates
    pc_right_loss = dice_loss(right_pc_true, right_pc_pred)

    # 3. Everything else
    normal_loss = dice_loss(y_true, y_pred)

    # Sum dice losses
    total_loss = n * (p_right_loss + pc_right_loss) + normal_loss
    return total_loss


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