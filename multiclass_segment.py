import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from unet import build_unet
from dice_loss_function import dice_loss
import tensorflow as tf
import numpy as np


def load_data(image_file_tensor, mask_file_tensor):
    # Convert symbolic tensors to strings
    image_file = tf.strings.reduce_join(image_file_tensor)
    mask_file = tf.strings.reduce_join(mask_file_tensor)

    # Decode strings to UTF-8
    image_file = tf.strings.unicode_decode(image_file, 'UTF-8')
    mask_file = tf.strings.unicode_decode(mask_file, 'UTF-8')

    # Load data from files
    image_data = tf.numpy_function(np.load, [image_file], tf.float32)['arr_0']
    mask_data = tf.numpy_function(np.load, [mask_file], tf.float32)['arr_0']

    return image_data, mask_data


def generate_dataset(batch_size):
    # Image Directories
    images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri"
    masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d"

    # Create a dataset of file paths
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    mask_files = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    # Create a tf.data.Dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))

    # Map the load_and_preprocess_data function to the dataset
    dataset = dataset.map(load_data)

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

    # Optionally, prefetch data for faster training
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    dropout_rate = 0.3
    epochs = 10

    # Build and compile model
    unet_model = build_unet(dropout_rate=dropout_rate)
    unet_model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy', 'loss'])

    # Get training dataset
    dataset = generate_dataset(batch_size=batch_size)

    # Train model
    unet_model.fit(dataset, epochs=epochs)
