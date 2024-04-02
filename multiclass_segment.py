import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

from unet import build_unet
from dice_loss_function import dice_loss
from create_dataset import get_dataset

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    dropout_rate = 0.3
    epochs = 10

    # Build and compile model
    unet_model = build_unet(dropout_rate=dropout_rate)
    unet_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', 'loss'])

    # Get datasets
    train_dataset = get_dataset(batch_size=batch_size, dataset_type='train')
    val_dataset = get_dataset(batch_size=batch_size, dataset_type='val')
    test_dataset = get_dataset(batch_size=batch_size, dataset_type='test')

    # Train model
    unet_model.fit(train_dataset, epochs=epochs)
