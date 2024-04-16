import os
import datetime
import tensorflow as tf
import numpy as np
import pickle
import time

from unet import build_unet
from dice_loss_function import dice_loss
from get_data_path import get_data_path


def assemble_mri_tensor(list_of_files, data_path):
    img_list = list()
    for file in list_of_files:
        abs_path = data_path + '/' + file
        img = tf.io.read_file(abs_path)
        img = tf.image.decode_bmp(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img_list.append(img)
    tensor_out = tf.stack(img_list)


def assemble_mask_tensor(list_of_files, data_path):
    img_list = list()
    for file in list_of_files:
        abs_path = data_path + '/' + file
        img = tf.io.read_file(abs_path)
        img = tf.image.decode_bmp(img)

        img = tf.image.convert_image_dtype(img, tf.float32)
        img_list.append(img)
    tensor_out = tf.stack(img_list)


if __name__ == '__main__':
    # See if GPU is being used
    print("TensorFlow version:", tf.__version__)

    if tf.config.list_physical_devices("GPU"):
        print("GPU is Available")
    else:
        print("No GPU detected")

    central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()

    with central_storage_strategy.scope():
        start_path = get_data_path()

        mri_path = start_path + "/train/mri"
        label_path = start_path + "/train/mask"

        mri = os.listdir(mri_path)
        mask = os.listdir(label_path)

        mri.sort()
        mask.sort()

        # Train/test split
        X_train_files = mri[:720]
        y_train_files = mask[:720]

        X_test_files = mri[720:840]
        y_test_files = mask[720:840]

        X_train = assemble_mri_tensor(X_train_files, mri_path)
        y_train = assemble_mask_tensor(y_train_files, label_path)
        X_test = assemble_mri_tensor(X_test_files, mri_path)
        y_test = assemble_mask_tensor(y_test_files, label_path)