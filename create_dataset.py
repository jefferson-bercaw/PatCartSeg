import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_image_and_label(image_file, label):
    # Load and preprocess the image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=1)  # Adjust channels as per your data
    # Perform any additional preprocessing as needed

    # Convert label to tensor (assuming it's already in the correct format)
    label = tf.convert_to_tensor(label)

    return image, label


def process_paths(two_paths):
    return tf.io.read_file(two_paths[0]), tf.io.read_file(two_paths[1])


def parse_image(filename):
    # Load the npz file
    image_npz = np.load(filename)
    # Extract the image array from the npz file
    image = image_npz['image']
    # Normalize the pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    return image


def parse_mask(filename):
    # Load the npz file
    mask_npz = np.load(filename)
    # Extract the mask array from the npz file
    mask = mask_npz['mask']
    # Normalize the pixel values to [0, 1]
    mask = mask.astype(np.float32) / 255.0
    return mask


def create_dataset(image_files, mask_files, batch_size=32, shuffle=True):
    # Create dataset from list of image files
    image_dataset = tf.data.Dataset.from_tensor_slices(image_files)
    # Map the parse_image function to decode and preprocess images
    image_dataset = image_dataset.map(lambda x: tf.numpy_function(parse_image, [x], tf.float32), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Create dataset from list of mask files
    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_files)
    # Map the parse_mask function to decode and preprocess masks
    mask_dataset = mask_dataset.map(lambda x: tf.numpy_function(parse_mask, [x], tf.float32), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Combine the image and mask datasets
    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))

    # Shuffle and batch the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_files))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def get_dataset(batch_size):
    images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri"
    masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d"

    # Create a dataset of file paths
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    label_files = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    dataset = create_dataset(image_files, label_files, batch_size=32, shuffle=True)


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri"
    masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d"

    # Create a dataset of file paths
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    label_files = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    print("Starting Creation of Dataset")
    dataset = create_dataset(image_files, label_files, batch_size=32, shuffle=True)

    # list_mri = tf.data.Dataset.list_files(str(images_dir + '/*'), shuffle=False, )
    # for f in list_mri.take(5):
    #     print(f.numpy())
    #
    # list_label = tf.data.Dataset.list_files(str(masks_dir + '/*'), shuffle=False)
    # for f in list_label.take(5):
    #     print(f.numpy())
    #
    # combined_paths = []
    # for mri, label in zip(list_mri, list_label):
    #     combined_paths.append([mri, label])
    #
    # pause()
