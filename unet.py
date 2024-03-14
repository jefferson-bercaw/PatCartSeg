import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

## Model Creation
def double_conv_block(lyr, n_filt, kernel_size):
    lyr = layers.Conv2D(n_filt, kernel_size, padding="same", activation="relu", kernel_initializer="he_normal")(lyr)
    lyr = layers.Conv2D(n_filt, kernel_size, padding="same", activation="relu", kernel_initializer="he_normal")(lyr)

    return lyr


def downsample_block(lyr, n_filt, kernel_size):
    f = double_conv_block(lyr, n_filt, kernel_size)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(lyr, conv_features, n_filt, kernel_size):
    lyr = layers.Conv2DTranspose(n_filt, kernel_size, 2, padding="same")(lyr)

    lyr = layers.concatenate([lyr, conv_features])

    lyr = layers.Dropout(0.3)(lyr)

    lyr = double_conv_block(lyr, n_filt, kernel_size)(lyr)
    return layer