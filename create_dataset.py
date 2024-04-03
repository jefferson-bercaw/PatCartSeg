import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np


def assemble_3d_mask(mask_2d, xy_dim):
    mask = np.zeros((xy_dim, xy_dim, 3))

    bckgrnd = np.zeros((xy_dim, xy_dim))
    p = np.zeros((xy_dim, xy_dim))
    pc = np.zeros((xy_dim, xy_dim))

    background_inds = mask_2d == 0
    bckgrnd[background_inds] = 1

    p_mask_inds = mask_2d == 1
    p[p_mask_inds] = 1

    pc_mask_inds = mask_2d == 2
    pc[pc_mask_inds] = 1

    mask[:, :, 0] = bckgrnd
    mask[:, :, 1] = p
    mask[:, :, 2] = pc

    return mask


def load_data(image_path, mask_path):
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.image.decode_bmp(image)
    mask = tf.image.decode_bmp(mask)

    mask_3d = assemble_3d_mask(mask, 512)

    image = tf.cast(image, tf.float64) / 255.0
    mask_3d = tf.cast(mask_3d, tf.float64)

    return image, mask_3d


def create_dataset(image_dir, mask_dir):
    # Create dataset from list of image files
    mri = tf.data.Dataset.list_files(image_dir, shuffle=False)
    mask = tf.data.Dataset.list_files(mask_dir, shuffle=False)

    dataset = tf.data.Dataset.zip((mri, mask))

    dataset = dataset.map(load_data)

    return dataset


def get_dataset(batch_size, dataset_type):
    if dataset_type == 'train':
        images_dir = "../Split_Data_BMP2/train/mri"
        mask_dir = "../Split_Data_BMP2/train/mask"
    elif dataset_type == 'test':
        images_dir = "../Split_Data_BMP2/test/mri"
        mask_dir = "../Split_Data_BMP2/test/mask"
    elif dataset_type == 'val' or dataset_type == 'validation':
        images_dir = "../Split_Data_BMP2/val/mri"
        mask_dir = "../Split_Data_BMP2/val/mask"
    else:
        raise(ValueError, f"The value {dataset_type} for the variable dataset_type is not one of 'train', 'test', "
                          f"or 'val'")

    # Add bmp wildcard
    images_dir = images_dir + "/*.bmp"
    mask_dir = mask_dir + "/*.bmp"

    # Create dataset
    dataset = create_dataset(images_dir, mask_dir)

    return dataset


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    dataset = get_dataset(batch_size=batch_size, dataset_type='train')

    for mri, mask in dataset:
        print(mri, mask)
        break
