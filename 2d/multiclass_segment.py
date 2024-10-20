import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import tensorflow as tf
import numpy as np
import pickle
import argparse
import pandas as pd
import sys

# Add the main directory to the system path
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

from unet import build_unet
from dice_loss_function import dice_loss
from create_dataset import get_dataset

parser = argparse.ArgumentParser(description="Training Options")
parser.add_argument("--dataset", type=str, default="cHTCO-Group", help="Dataset to train on.")
parser.add_argument("--tissue", type=str, default='p', help="Tissue type to segment. Choose 'p' for patella or 'c' for patellar cartilage.")
parser.add_argument("--learningrate", type=float, default=0.0001, help="Initial learning rate for Adam optimizer.")
parser.add_argument("--batch", type=int, default=32, help="Batch size for training.")
parser.add_argument("--depth", type=int, default=4, help="Depth of U-Net model.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for U-Net model.")
parser.add_argument("--kernel", type=int, default=3, help="Kernel size for convolutional layers.")
parser.add_argument("--epochs", type=int, default=500, help="Maximum number of epochs to train for.")
args = parser.parse_args()


def save_model_info(model_info):
    """Write to an excel spreadsheet that already exists"""
    df = pd.DataFrame([model_info])
    folder_path = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(folder_path, os.pardir))
    with pd.ExcelWriter(os.path.join(parent_dir, "results", "2d_gridsearch.xlsx"), mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, index=False, header=False, startrow=writer.sheets["Sheet1"].max_row)


if __name__ == "__main__":

    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    print("Starting training script code")
    # Main directory
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Hyperparameters
        batch_size = args.batch
        model_depth = args.depth
        dropout_rate = args.dropout
        epochs = args.epochs
        dataset_name = args.dataset
        patience = args.epochs // 2
        min_delta = 0.001
        initial_learning_rate = args.learningrate
        kernel_size = args.kernel

        # Build and compile model
        unet_model = build_unet(model_depth=model_depth, dropout_rate=dropout_rate, kernel_size=3)
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

        unet_model.compile(optimizer=adam_optimizer,
                           loss=dice_loss,
                           metrics=[tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.FalsePositives(thresholds=0.5, name='FP'),
                                    tf.keras.metrics.FalseNegatives(thresholds=0.5, name='FN'),
                                    tf.keras.metrics.TruePositives(thresholds=0.5, name='TP'),
                                    tf.keras.metrics.TrueNegatives(thresholds=0.5, name='TN')])

        # Get datasets
        train_dataset = get_dataset(dataset_name=dataset_name, dataset_type="train", batch_size=batch_size, tissue=parser.parse_args().tissue)
        val_dataset = get_dataset(dataset_name=dataset_name, dataset_type="val", batch_size=batch_size, tissue=parser.parse_args().tissue)

        # Early stopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   patience=patience,
                                                                   min_delta=min_delta,
                                                                   verbose=1)

        class PerformanceCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super(PerformanceCallback, self).__init__()
                self.best_val_loss = float('inf')

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                val_loss = logs.get('val_loss')

                print(f"Epoch {epoch + 1} - Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")

                if val_loss is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        print("Saving model!")
                        self.model.save(os.path.join(main_dir, "models", f"unet2d-{parser.parse_args().tissue}_8888-88-88_88-88-88_{dataset_name}.h5"))

        print("Training!")
        # Train model
        history = unet_model.fit(train_dataset,
                                 epochs=epochs,
                                 callbacks=[early_stopping_callback, PerformanceCallback()],
                                 validation_data=val_dataset,
                                 verbose=2)

        # Save model
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"unet2d-{parser.parse_args().tissue}_{current_time}_{dataset_name}"
        unet_model.save(os.path.join(main_dir, "models", f"{model_name}.h5"))

        # Save history
        hist_name = f"{model_name}.pkl"
        with open(os.path.join(main_dir, "history", hist_name), "wb") as f:
            pickle.dump(history.history, f)

        # Print saving model
        print(f"Saving model to {model_name}.h5")
        print(f"Model Parameters:\n"
              f"patience: {patience}\n"
              f"learning_rate: {initial_learning_rate}\n"
              f"kernel_size: {kernel_size}\n"
              f"batch_size: {batch_size}\n"
              f"dropout_rate: {dropout_rate}\n"
              f"max epochs: {epochs}\n"
              f"epochs trained for: {len(history.history['loss'])}\n"
              f"model depth: {model_depth}\n")

        model_info = {"model_name": model_name,
                      "patience": patience,
                      "batch_size": batch_size,
                      "kernel_size": kernel_size,
                      "learning_rate": initial_learning_rate,
                      "dropout_rate": dropout_rate,
                      "max_epochs": epochs,
                      "epochs_trained_for": len(history.history['loss']),
                      "model_depth": model_depth}

        save_model_info(model_info)