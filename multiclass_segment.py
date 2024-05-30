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
parser.add_argument("-a", "--arr", help="Enter the suffix of the dataset we're testing", type=int)
args = parser.parse_args()


def array_to_metric(arr_num):
    dropout = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    return dropout[arr_num]


if __name__ == "__main__":
    # GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Hyperparameters
        batch_size = 20
        dropout_rate = array_to_metric(args.arr)
        epochs = 500
        patience = 20
        min_delta = 0.0001

        dataset = f"cHT5_{int(args.arr)}"

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
        train_dataset = get_dataset(batch_size=batch_size, dataset_type='train', dataset=dataset)
        val_dataset = get_dataset(batch_size=batch_size, dataset_type='val', dataset=dataset)

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
        model_name = f"unet_{current_time}_{dataset}"
        unet_model.save(os.path.join("models", f"{model_name}.h5"))

        # Print saving model
        print(f"Saving model to {model_name}.h5")

        # Save history
        hist_name = f"{model_name}.pkl"
        with open(os.path.join("history", hist_name), "wb") as f:
            pickle.dump(history, f)
