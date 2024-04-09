import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from get_data_path import get_data_path
import matplotlib.pyplot as plt


def assemble_3d_mask(mask_2d, xy_dim):

    # background_inds = tf.equal(mask_2d, 0)
    # bckgrnd = tf.where(background_inds, 1, 0)

    p_mask_inds = tf.equal(mask_2d, 1)
    p = tf.where(p_mask_inds, 1, 0)

    pc_mask_inds = tf.equal(mask_2d, 2)
    pc = tf.where(pc_mask_inds, 1, 0)

    mask = tf.stack([p, pc], axis=-1)
    mask = tf.squeeze(mask, axis=-2)

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

    data_path = get_data_path()

    if dataset_type == 'train':
        images_dir = data_path + "/train/mri"
        mask_dir = data_path + "/train/mask"
    elif dataset_type == 'test':
        images_dir = data_path + "/test/mri"
        mask_dir = data_path + "/test/mask"
    elif dataset_type == 'val' or dataset_type == 'validation':
        images_dir = data_path + "/val/mri"
        mask_dir = data_path + "/val/mask"
    else:
        raise(ValueError, f"The value {dataset_type} for the variable dataset_type is not one of 'train', 'test', "
                          f"or 'val'")

    # Add bmp wildcard
    images_dir = images_dir + "/*.bmp"
    mask_dir = mask_dir + "/*.bmp"

    # Create dataset
    dataset = create_dataset(images_dir, mask_dir)

    dataset = dataset.cache()

    # randomly shuffle
    dataset = dataset.shuffle(buffer_size=tf.data.experimental.cardinality(dataset).numpy() // 2, seed=42)

    # Prefetch batch into memory at a given time
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Parallelize Data Loading Step
    dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)

    # Batch each dataset
    dataset = dataset.batch(batch_size=batch_size)

    # Cache dataset into memory on first epoch
    dataset = dataset.cache()

    return dataset


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    dataset = get_dataset(batch_size=batch_size, dataset_type='train')

    # for mri, mask in dataset:
    #     print(mri, mask)
    #     plt.imshow(mri, cmap='gray')
    #     plt.show()
    #
    #     plt.imshow(mask[:, :, 0], cmap='gray')
    #     plt.show()
    #
    #     plt.imshow(mask[:, :, 1], cmap='gray')
    #     plt.show()
    #
    #     plt.imshow(mask[:, :, 2], cmap='gray')
    #     plt.show()
    #
    #     pause = input("Enter to continue")

