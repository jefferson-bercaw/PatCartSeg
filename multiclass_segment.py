import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import tensorflow as tf
import numpy as np
import pickle
import time
import argparse

from unet import build_unet
from dice_loss_function import dice_loss
from create_dataset import get_dataset


parser = argparse.ArgumentParser(description="Training Options")
parser.add_argument("-d", "--dataset", help="Enter the suffix of the dataset we're testing")
args = parser.parse_args()
print(f"Dataset argument: {args.dataset}")


class RecordHistory(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset):
        self.validation_dataset = validation_dataset
        self.history = {'loss': [], 'val_loss': []}

    def on_batch_end(self, batch, logs=None):
        # Calculate dice score for training data
        self.history['loss'].append(logs['loss'])

    def on_epoch_end(self, epoch, logs=None):
        val_loss = self.model.evaluate(self.validation_dataset, verbose=0)
        self.history['val_loss'].append(val_loss)


if __name__ == "__main__":
    # GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Hyperparameters
        batch_size = 20
        dropout_rate = 0.3
        epochs = 500
        patience = 40
        min_delta = 0.0001

        # Build and compile model
        unet_model = build_unet(dropout_rate=dropout_rate)

        unet_model.compile(optimizer='adam',
                           loss=dice_loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.FalsePositives(thresholds=0.5, name='FP'),
                                    tf.keras.metrics.FalseNegatives(thresholds=0.5, name='FN'),
                                    tf.keras.metrics.TruePositives(thresholds=0.5, name='TP'),
                                    tf.keras.metrics.TrueNegatives(thresholds=0.5, name='TN')])

        # Get datasets
        train_dataset = get_dataset(batch_size=batch_size, dataset_type='train', dataset=args.dataset)
        val_dataset = get_dataset(batch_size=batch_size, dataset_type='val', dataset=args.dataset)

        # Early stopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   patience=patience,
                                                                   min_delta=min_delta,
                                                                   verbose=1)

        checkpoint_filepath = os.path.join("checkpoints", "tmp", "checkpoint")

        # Define model callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True,
                                                         save_weights_only=True)

        # Initialize recording history
        record_history_callback = RecordHistory(validation_dataset=val_dataset)

        # Train model
        history = unet_model.fit(train_dataset,
                                 epochs=epochs,
                                 callbacks=[cp_callback, record_history_callback, early_stopping_callback],
                                 validation_data=val_dataset)

        # Save model
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"unet_{current_time}_{args.dataset}"
        unet_model.save(os.path.join("models", f"{model_name}.h5"))

        # Save best model
        unet_model.load_weights(checkpoint_filepath)
        unet_model.save(os.path.join("models", f"lowest_val_loss_{model_name}.h5"))

        # Save history
        hist_name = f"{model_name}.pkl"
        with open(os.path.join("history", hist_name), "wb") as f:
            pickle.dump(history.history, f)
