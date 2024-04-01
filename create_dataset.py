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


def image_generator(image_paths, batch_size):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_images = []
            for path in image_paths[i:i + batch_size]:
                # Load and preprocess images
                image = np.load(path)
                # Perform any necessary preprocessing here
                batch_images.append(image)
            yield np.array(batch_images)


def label_generator(label_paths, batch_size):
    while True:
        for i in range(0, len(label_paths), batch_size):
            batch_labels = []
            for path in label_paths[i:i+batch_size]:
                # Load and preprocess labels
                label = np.load(path)
                # Perform any necessary preprocessing here
                batch_labels.append(label)
            yield np.array(batch_labels)


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri"
    masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d"

    # Create a dataset of file paths
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    label_files = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    image_data_generator = ImageDataGenerator(rescale=1./255)
    label_data_generator = ImageDataGenerator()

    image_generator = image_data_generator.flow(image_generator(image_files, batch_size=batch_size),
                                                batch_size=batch_size)
    label_generator = label_data_generator.flow(label_generator(label_files, batch_size=batch_size),
                                                batch_size=batch_size)
    train_generator = zip(image_generator, label_generator)
