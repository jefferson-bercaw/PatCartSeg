import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_image_and_label(image_file, label):
    # Load and preprocess the image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=1)  # Adjust channels as per your data
    # Perform any additional preprocessing as needed

    # Convert label to tensor (assuming it's already in the correct format)
    label = tf.convert_to_tensor(label)

    return image, label


if __name__ == '__main__':

    images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri"
    masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d"

    # Create a dataset of file paths
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    labels = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    dataset = dataset.map(load_image_and_label)


