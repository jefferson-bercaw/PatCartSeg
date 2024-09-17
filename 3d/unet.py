import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Model Creation
def double_conv_block(lyr, n_filt, kernel_size):
    lyr = layers.Conv3D(n_filt, kernel_size, padding="same", kernel_initializer="he_normal")(lyr)
    lyr = tf.keras.activations.relu(lyr)

    lyr = layers.Conv3D(n_filt, kernel_size, padding="same", kernel_initializer="he_normal")(lyr)
    lyr = tf.keras.activations.relu(lyr)

    return lyr


def downsample_block(lyr, n_filt, kernel_size, dropout_rate):
    f = double_conv_block(lyr, n_filt, kernel_size)
    p = layers.MaxPool3D(2)(f)
    p = layers.Dropout(dropout_rate)(p)
    return f, p


def upsample_block(lyr, conv_features, n_filt, kernel_size, dropout_rate):
    lyr = layers.Conv3DTranspose(n_filt, kernel_size, 2, padding="same")(lyr)
    if lyr.shape[3] != conv_features.shape[3]:
        to_add = conv_features.shape[3] - lyr.shape[3]
        lyr = layers.ZeroPadding3D(padding=((0, 0), (0, 0), (0, to_add)))(lyr)
    lyr = layers.concatenate([lyr, conv_features])
    lyr = layers.Dropout(dropout_rate)(lyr)
    lyr = double_conv_block(lyr, n_filt, kernel_size)
    return lyr


def build_unet(model_depth, dropout_rate, kernel_size):
    # inputs
    inputs = layers.Input(shape=(224, 128, 56, 1))  # Adjusted input shape

    if model_depth == 3:
        start_filt = 128

        # encoder
        f1, p1 = downsample_block(inputs, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        print(f1.shape)
        f2, p2 = downsample_block(p1, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        print(f2.shape)
        # bottleneck
        bottleneck = double_conv_block(p2, n_filt=4 * start_filt, kernel_size=kernel_size)
        print(bottleneck.shape)
        # decoder
        u4 = upsample_block(bottleneck, conv_features=f2, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        print(u4.shape)
        u5 = upsample_block(u4, conv_features=f1, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        print(u5.shape)

        outputs = layers.Conv3D(filters=2, kernel_size=1, padding="same", activation="sigmoid")(u5)  # Output shape: (256, 256, 70, 2)

    elif model_depth == 4:
        start_filt = 64

        # encoder
        f1, p1 = downsample_block(inputs, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f2, p2 = downsample_block(p1, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f3, p3 = downsample_block(p2, n_filt=4 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)

        # bottleneck
        bottleneck = double_conv_block(p3, n_filt=8 * start_filt, kernel_size=kernel_size)

        # decoder
        u5 = upsample_block(bottleneck, conv_features=f3, n_filt=4 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u6 = upsample_block(u5, conv_features=f2, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u7 = upsample_block(u6, conv_features=f1, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        outputs = layers.Conv3D(filters=1, kernel_size=1, padding="same", activation="sigmoid")(u7)

    elif model_depth == 5:
        start_filt = 32

        # encoder
        f1, p1 = downsample_block(inputs, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f2, p2 = downsample_block(p1, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f3, p3 = downsample_block(p2, n_filt=4 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f4, p4 = downsample_block(p3, n_filt=8 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)

        # bottleneck
        bottleneck = double_conv_block(p4, n_filt=16 * start_filt, kernel_size=kernel_size)

        # decoder
        u6 = upsample_block(bottleneck, conv_features=f4, n_filt=8 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u7 = upsample_block(u6, conv_features=f3, n_filt=4 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u8 = upsample_block(u7, conv_features=f2, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u9 = upsample_block(u8, conv_features=f1, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        outputs = layers.Conv3D(filters=1, kernel_size=1, padding="same", activation="sigmoid")(u9)

    elif model_depth == 6:
        start_filt = 16

        # encoder
        f1, p1 = downsample_block(inputs, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f2, p2 = downsample_block(p1, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f3, p3 = downsample_block(p2, n_filt=4 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f4, p4 = downsample_block(p3, n_filt=8 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        f5, p5 = downsample_block(p4, n_filt=16 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)

        # bottleneck
        bottleneck = double_conv_block(p5, n_filt=32 * start_filt, kernel_size=kernel_size)

        # decoder
        u7 = upsample_block(bottleneck, conv_features=f5, n_filt=16 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u8 = upsample_block(u7, conv_features=f4, n_filt=8 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u9 = upsample_block(u8, conv_features=f3, n_filt=4 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u10 = upsample_block(u9, conv_features=f2, n_filt=2 * start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        u11 = upsample_block(u10, conv_features=f1, n_filt=start_filt, kernel_size=kernel_size, dropout_rate=dropout_rate)
        outputs = layers.Conv3D(filters=1, kernel_size=1, padding="same", activation="sigmoid")(u11)

    # outputs
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model


if __name__ == "__main__":
    unet_model = build_unet(model_depth=4, dropout_rate=0.1, kernel_size=5)
    unet_model.summary()