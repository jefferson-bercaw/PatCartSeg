import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import tensorflow as tf
import numpy as np
import pickle
import time

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
    # GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Hyperparameters
        batch_size = 32
        dropout_rate = 0.3
        epochs = 500
        patience = 10
        min_delta = 0.0001

        # Build and compile model
        t = time.time()
        unet_model = build_unet(dropout_rate=dropout_rate)
        print(f"It took {time.time() - t} seconds to build the UNet")

        t = time.time()
        unet_model.compile(optimizer='adam',
                           loss=dice_loss,
                           metrics=['accuracy',
                                    tf.keras.metrics.FalsePositives(thresholds=0.5, name='FP'),
                                    tf.keras.metrics.FalseNegatives(thresholds=0.5, name='FN'),
                                    tf.keras.metrics.TruePositives(thresholds=0.5, name='TP'),
                                    tf.keras.metrics.TrueNegatives(thresholds=0.5, name='TN')])
        print(f"It took {time.time() - t} seconds to compile the UNet")

        # Get datasets
        t = time.time()
        train_dataset = get_dataset(batch_size=batch_size, dataset_type='train')
        print(f"It took {time.time() - t} seconds to get the training dataset")

        t = time.time()
        val_dataset = get_dataset(batch_size=batch_size, dataset_type='val')
        print(f"It took {time.time() - t} seconds to get the val dataset")

        # Iterate over the dataset to cache it into memory
        t = time.time()
        for _ in train_dataset:
            pass
        print(f"It took {time.time() - t} seconds to loop through the training dataset")

        t = time.time()
        for _ in val_dataset:
            pass
        print(f"It took {time.time() - t} seconds to loop through the validation dataset")

        # Early stopping callback
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   patience=patience,
                                                                   min_delta=min_delta,
                                                                   verbose=1)

        # Define model callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./models/unet_04102024.keras",
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=True)

        # Initialize recording history
        record_history_callback = RecordHistory(validation_dataset=val_dataset)

        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(epochs):
            for batch in train_dataset:
                mri, y_true = batch
                with tf.GradientTape() as tape:
                    y_pred = unet_model(mri)
                    loss = dice_loss(y_true, y_pred)

                gradients = tape.gradient(loss, unet_model.trainable_variables)

                optimizer.apply_gradients(zip(gradients, unet_model.trainable_variables))

        # Train model
        # history = unet_model.fit(train_dataset,
        #                          epochs=epochs,
        #                          callbacks=[cp_callback, record_history_callback, early_stopping_callback],
        #                          validation_data=val_dataset)

        # Save model
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = f"unet_{current_time}.h5"
        unet_model.save(f"./models/{model_name}")

        # Save history
        hist_name = f"unet_{current_time}.pkl"
        with open(f"./history/{hist_name}", "wb") as f:
            pickle.dump(history.history, f)
