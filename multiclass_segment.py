import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import tensorflow as tf
import numpy as np
import pickle

from unet import build_unet
from dice_loss_function import dice_loss
from create_dataset import get_dataset


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

    # Define model callbacks
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="{epoch:02d}-{val_loss:.4f}.keras",
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True)

    # Train model
    history = unet_model.fit(train_dataset,
                             epochs=epochs,
                             callbacks=[cp_callback],
                             validation_data=val_dataset)

    # Save model
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"unet_{current_time}.h5"
    unet_model.save(f"./models/{model_name}")

    # Save history
    hist_name = f"unet_{current_time}.pkl"
    with open(f"./history/{hist_name}", "wb") as f:
        pickle.dump(history.history, f)
