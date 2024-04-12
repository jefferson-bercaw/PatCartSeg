import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime
import tensorflow as tf
import numpy as np
import pickle
import time

from unet import build_unet
from dice_loss_function import dice_loss
from get_data_path import get_data_path


def assemble_tensor(list_of_files, beginning_path):
    img_list = list()
    for file in list_of_files:
        abs_path = beginning_path + '/' + file
        img = tf.io.read_file(abs_path)
        img = tf.image.decode_bmp(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img_list.append(img)
    tensor_out = tf.stack(img_list)
    return tensor_out


def convert_mask(mask_tensor):
    mask_P = tf.cast(tf.abs(mask_tensor - 0.003) < 0.002, tf.float32)
    mask_PC = tf.cast(tf.abs(mask_tensor - 0.007) < 0.002, tf.float32)
    output_tensor = tf.concat([mask_P, mask_PC], axis=-1)
    return output_tensor


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

        X_train = assemble_tensor(X_train_files, mri_path)
        y_train = convert_mask(assemble_tensor(y_train_files, label_path))
        X_test = assemble_tensor(X_test_files, mri_path)
        y_test = convert_mask(assemble_tensor(y_test_files, label_path))

        # Training
        batch_size = 32
        dropout_rate = 0.3
        epochs = 500
        patience = 10
        min_delta = 0.001

        unet_model = build_unet(dropout_rate=dropout_rate)
        unet_model.compile(optimizer='adam',
                           loss=dice_loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.FalsePositives(thresholds=0.5, name='FP'),
                                    tf.keras.metrics.FalseNegatives(thresholds=0.5, name='FN'),
                                    tf.keras.metrics.TruePositives(thresholds=0.5, name='TP'),
                                    tf.keras.metrics.TrueNegatives(thresholds=0.5, name='TN')])
        history = unet_model.fit(X_train, y_train,
                                 epochs=epochs)
        unet_model.save(filename="Small_Unet.keras")
