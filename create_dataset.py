import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from get_data_path import get_data_path
import matplotlib.pyplot as plt
from scipy import ndimage
from glob import glob


def assemble_4d_mask(mask_3d):
    """Assembles a 4D tf.Tensor of 1s and 0s pertaining to the patella and patellar cartilage

    Inputs: mask_3d (ndarray) of size (xy_dim, xy_dim, depth) containing 0, 1, and 2 pertaining to background, patella, and patellar cartilage, respectively

    Outputs: mask_4d (tf.tensor) of size (xy_dim, xy_dim, depth, 2) of 1s and 0s corresponding to P and PC
    """
    p_mask_inds = tf.equal(mask_3d, 1)
    p = tf.where(p_mask_inds, 1, 0)

    pc_mask_inds = tf.equal(mask_3d, 2)
    pc = tf.where(pc_mask_inds, 1, 0)

    mask = tf.stack([p, pc], axis=-1)
    mask = tf.squeeze(mask, axis=-2)

    return mask


def load_images(mri_folder, mask_folder):

    mri_files = tf.io.matching_files(tf.strings.join([mri_folder, '/*.bmp']))
    mask_files = tf.io.matching_files(tf.strings.join([mask_folder, '/*.bmp']))

    mris = tf.map_fn(lambda img_file: tf.image.decode_bmp(tf.io.read_file(img_file)), mri_files, fn_output_signature=tf.uint8)
    masks = tf.map_fn(lambda img_file: tf.image.decode_bmp(tf.io.read_file(img_file)), mask_files, fn_output_signature=tf.uint8)

    mri_3d = tf.cast(mris, tf.float64) / 255.0
    mask_4d = assemble_4d_mask(masks)
    mask_4d = tf.cast(mask_4d, tf.float64)

    mri_3d = tf.transpose(mri_3d, perm=[1, 2, 0, 3])
    mask_4d = tf.transpose(mask_4d, perm=[1, 2, 0, 3])

    return mri_3d, mask_4d


def visualize_dataset(dataset, num_samples=5):
    for mri_images, mask_images in dataset.take(1):  # Take one batch
        # Visualize a few samples
        for i in range(min(num_samples, mri_images.shape[3])):
            plt.figure(figsize=(12, 6))

            # Original MRI Image
            plt.subplot(1, 2, 1)
            plt.title("MRI Image")
            plt.imshow(mri_images[0, :, :, i], cmap='gray')  # Assuming grayscale
            plt.axis('off')

            # Corresponding Mask
            plt.subplot(1, 2, 2)
            plt.title("Mask Image")
            plt.imshow(mask_images[0, :, :, i, 0] * 100, cmap='gray')  # Assuming grayscale
            plt.axis('off')

            plt.show()
            plt.savefig("test.png")


def get_dataset(batch_size, dataset_type, dataset):
    """Returns a tf.data.Dataset object given batch_size, dataset_type, and dataset selection

    Inputs: batch_size: (int), batch size for dataset
            dataset_type: one of ["train", "test", "val"] corresponding to the dataset type we're creating
            dataset: dataset flag from argparse calling, currently should be CHT-Group

    Outputs: dataset: tf.data.Dataset object containing filenames and mapping function for calling
    """

    data_path = get_data_path(dataset)

    # MRI
    mri_path = os.path.join(data_path, dataset_type, "mri")
    mri_folders = sorted(
        [os.path.join(mri_path, d) for d in os.listdir(mri_path) if os.path.isdir(os.path.join(mri_path, d))])

    # Mask
    mask_path = os.path.join(data_path, dataset_type, "mask")
    mask_folders = sorted(
        [os.path.join(mask_path, d) for d in os.listdir(mask_path) if os.path.isdir(os.path.join(mask_path, d))])

    mri_folders_ds = tf.data.Dataset.from_tensor_slices(mri_folders)
    mask_folders_ds = tf.data.Dataset.from_tensor_slices(mask_folders)

    dataset = tf.data.Dataset.zip((mri_folders_ds, mask_folders_ds))
    dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)

    # randomly shuffle if training
    if dataset_type == "train":
        dataset = dataset.shuffle(buffer_size=min([tf.data.experimental.cardinality(dataset).numpy() // 4, 100]), seed=42)

    # Parallelize Data Loading Step
    # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)

    # Batch each dataset
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch batch into memory at a given time
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Cache dataset to memory
    # dataset = dataset.cache()

    return dataset


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 4
    dataset = get_dataset(batch_size=batch_size, dataset_type='val', dataset="CHT-Group")
    visualize_dataset(dataset, num_samples=5)
    iterable = iter(dataset)
    out = next(iterable)
    mri, label = out

    print(f"MRI size: {mri.numpy().shape}")
    print(f"Mask size: {label.numpy().shape}")
