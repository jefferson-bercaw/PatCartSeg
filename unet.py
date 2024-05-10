import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Model Creation
def double_conv_block(lyr, n_filt, kernel_size):
    lyr = layers.Conv2D(n_filt, kernel_size, padding="same", kernel_initializer="he_normal")(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = tf.keras.activations.relu(lyr)

    lyr = layers.Conv2D(n_filt, kernel_size, padding="same", kernel_initializer="he_normal")(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = tf.keras.activations.relu(lyr)

    return lyr


def downsample_block(lyr, n_filt, kernel_size, dropout_rate):
    f = double_conv_block(lyr, n_filt, kernel_size)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout_rate)(p)

    return f, p


def upsample_block(lyr, conv_features, n_filt, kernel_size, dropout_rate):
    lyr = layers.Conv2DTranspose(n_filt, kernel_size, 2, padding="same")(lyr)

    lyr = layers.concatenate([lyr, conv_features])

    lyr = layers.Dropout(dropout_rate)(lyr)

    lyr = double_conv_block(lyr, n_filt, kernel_size)
    return lyr


def build_unet(dropout_rate):

    # inputs
    inputs = layers.Input(shape=(512, 512, 1))
    start_filt = 32

    # encoder
    f1, p1 = downsample_block(inputs, n_filt=start_filt, kernel_size=3, dropout_rate=dropout_rate)
    f2, p2 = downsample_block(p1, n_filt=2*start_filt, kernel_size=3, dropout_rate=dropout_rate)
    f3, p3 = downsample_block(p2, n_filt=4*start_filt, kernel_size=3, dropout_rate=dropout_rate)
    f4, p4 = downsample_block(p3, n_filt=8*start_filt, kernel_size=3, dropout_rate=dropout_rate)

    # bottleneck
    bottleneck = double_conv_block(p4, n_filt=16*start_filt, kernel_size=3)

    # decoder
    u6 = upsample_block(bottleneck, conv_features=f4, n_filt=8*start_filt, kernel_size=3, dropout_rate=dropout_rate)
    u7 = upsample_block(u6, conv_features=f3, n_filt=4*start_filt, kernel_size=3, dropout_rate=dropout_rate)
    u8 = upsample_block(u7, conv_features=f2, n_filt=2*start_filt, kernel_size=3, dropout_rate=dropout_rate)
    u9 = upsample_block(u8, conv_features=f1, n_filt=start_filt, kernel_size=3, dropout_rate=dropout_rate)

    # outputs
    outputs = layers.Conv2D(filters=2, kernel_size=1, padding="same", activation="sigmoid")(u9)
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model


if __name__ == "__main__":
    unet_model = build_unet(dropout_rate=0.3)
    unet_model.summary()
