import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import tensorflow as tf
import numpy as np
import pickle
import argparse

from unet import build_unet
from dice_loss_function import dice_loss
from create_dataset import get_dataset


parser = argparse.ArgumentParser(description="Training Options")

if __name__ == "__main__":

    # GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Hyperparameters
        batch_size = 2
        model_depth = 4
        dropout_rate = 0.1
        epochs = 2000
        patience = 500
        min_delta = 0.0001
        initial_learning_rate = 0.001

        dataset_name = "CHT-Group"

        # Build and compile model
        unet_model = build_unet(model_depth=model_depth)
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

        unet_model.compile(optimizer=adam_optimizer,
                           loss=dice_loss,
                           metrics=[tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.FalsePositives(thresholds=0.5, name='FP'),
                                    tf.keras.metrics.FalseNegatives(thresholds=0.5, name='FN'),
                                    tf.keras.metrics.TruePositives(thresholds=0.5, name='TP'),
                                    tf.keras.metrics.TrueNegatives(thresholds=0.5, name='TN')])

        # Get datasets
        train_dataset = get_dataset(dataset_name=dataset_name, dataset_type="train", batch_size=batch_size)
        val_dataset = get_dataset(dataset_name="CHT-Group", dataset_type="val", batch_size=batch_size)

        # Early stopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   patience=patience,
                                                                   min_delta=min_delta,
                                                                   verbose=1)

        # Train model
        history = unet_model.fit(train_dataset,
                                 epochs=epochs,
                                 callbacks=early_stopping_callback,
                                 validation_data=val_dataset)

        # Save model
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"unet3d_{current_time}_{dataset_name}"
        unet_model.save(os.path.join("models", f"{model_name}.h5"))

        # Save history
        hist_name = f"{model_name}.pkl"
        with open(os.path.join("history", hist_name), "wb") as f:
            pickle.dump(history.history, f)

        # Print saving model
        print(f"Saving model to {model_name}.h5")
        print(f"Model Parameters:"
              f"patience: {patience}"
              f"batch_size: {batch_size}"
              f"dropout_rate: {dropout_rate}"
              f"max epochs: {epochs}"
              f"epochs trained for: {len(history.history['loss'])}"
              f"model depth: {model_depth}")
