import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np


def load_data(image_path, mask_P_path, mask_PC_path):
    image = tf.io.read_file(image_path)
    mask_P = tf.io.read_file(mask_P_path)
    mask_PC = tf.io.read_file(mask_PC_path)

    image = image.astype(np.float64) / 255.0
    mask_P = mask_P.astype(np.float64)
    mask_PC = mask_PC.astype(np.float64)

    return image, mask_P, mask_PC


def create_dataset(image_dir, mask_P_dir, mask_PC_dir):
    # Create dataset from list of image files
    mri = tf.data.Dataset.list_files(image_dir, shuffle=False)
    mask_P = tf.data.Dataset.list_files(mask_P_dir, shuffle=False)
    mask_PC = tf.data.Dataset.list_files(mask_PC_dir, shuffle=False)

    dataset = tf.data.Dataset.zip((mri, mask_P, mask_PC))

    dataset = dataset.map(load_data)

    return dataset


def get_dataset(batch_size, dataset_type):
    if dataset_type == 'train':
        images_dir = "../Split_Data_BMP/train/mri"
        mask_P_dir = "../Split_Data_BMP/train/mask_P"
        mask_PC_dir = "../Split_Data_BMP/train/mask_PC"
    elif dataset_type == 'test':
        images_dir = "../Split_Data_BMP/test/mri"
        mask_P_dir = "../Split_Data_BMP/test/mask_P"
        mask_PC_dir = "../Split_Data_BMP/test/mask_PC"
    elif dataset_type == 'val' or dataset_type == 'validation':
        images_dir = "../Split_Data_BMP/val/mri"
        mask_P_dir = "../Split_Data_BMP/val/mask_P"
        mask_PC_dir = "../Split_Data_BMP/val/mask_PC"
    else:
        raise(ValueError, f"The value {dataset_type} for the variable dataset_type is not one of 'train', 'test', "
                          f"or 'val'")

    # Add bmp wildcard
    images_dir = images_dir + "/*.bmp"
    mask_P_dir = mask_P_dir + "/*.bmp"
    mask_PC_dir = mask_PC_dir + "/*.bmp"

    # Create dataset
    dataset = create_dataset(images_dir, mask_P_dir, mask_PC_dir)

    return dataset


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    dataset = get_dataset(batch_size=batch_size, 'train')

    # Create a dataset of file paths
    # image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    # label_files = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    dataset = create_dataset(images_dir, masks_dir)

    for mri, mask in dataset:
        print(mri, mask)
        break
