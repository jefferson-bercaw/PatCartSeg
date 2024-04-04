import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import tensorflow as tf
import numpy as np
import pickle

from unet import build_unet
from dice_loss_function import dice_loss
from create_dataset import get_dataset

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    dropout_rate = 0.3
    epochs = 5

    # Build and compile model
    unet_model = build_unet(dropout_rate=dropout_rate)
    unet_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

    # Get datasets
    train_dataset = get_dataset(batch_size=batch_size, dataset_type='train')
    val_dataset = get_dataset(batch_size=batch_size, dataset_type='val')
    test_dataset = get_dataset(batch_size=batch_size, dataset_type='test')

    # Train model
    history = unet_model.fit(train_dataset, epochs=epochs)

    # Save model
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"unet_{current_time}.h5"
    unet_model.save(f"./models/{model_name}")

    # Save history
    hist_name = f"unet_{current_time}.pkl"
    with open(f"./history/{hist_name}", "wb") as f:
        pickle.dump(history.history, f)
