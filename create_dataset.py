import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from get_data_path import get_data_path
import matplotlib.pyplot as plt
from scipy import ndimage


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
    # tf.print(mask.shape)

    return mask


def load_test_data(image_path, mask_path):
    """Load in MRI slice and mask slice for the test dataset given their absolute paths

    Inputs: image_path: absolute path of the .bmp file where the MRI slice is stored
            mask_path: absolute path of the .bmp file where the mask is stored

    Outputs: filename: filename of the .bmp file for this volume (e.g. AS_010-0011.bmp)
             image: tf.float64 normalized to [0, 1] of the MRI slice, size (xy_dim, xy_dim)
             mask_3d: tf.float64 of 0s and 1s of the mask, size (xy_dim, xy_dim, 2)
    """

    # Define vectors of bmp slices
    numbers = tf.range(start=1, limit=57, dtype=tf.int32)
    slice_nums = tf.strings.as_string(numbers, width=4, fill='0')

    # Remove .txt extension
    image_parts = tf.strings.split(image_path, sep='.')[0]
    mask_parts = tf.strings.split(mask_path, sep='.')[0]

    # Combine into list of strings and load in
    images = tf.strings.join([image_parts, "-", slice_nums, '.bmp'])
    masks = tf.strings.join([mask_parts, "-", slice_nums, '.bmp'])

    # tf.print(images)

    image = tf.map_fn(read_file, images, fn_output_signature=tf.uint8)
    mask = tf.map_fn(read_file, masks, fn_output_signature=tf.uint8)

    # tf.print(image.shape)

    image = tf.reshape(image, [224, 128, 56, 1])
    mask = tf.reshape(mask, [224, 128, 56, 1])

    mask_4d = assemble_4d_mask(mask)

    image = tf.cast(image, tf.float64) / 255.0
    mask_4d = tf.cast(mask_4d, tf.float64)

    return image_parts, image, mask_4d


def read_file(file_path):
    """Reads in a file from a given file path"""
    file_path_new = tf.strings.regex_replace(file_path, r"\\", r"/")

    content = tf.io.read_file(file_path_new)

    image = tf.io.decode_bmp(content, channels=0)
    image = tf.reshape(image, [224, 128, 1])

    return image


def load_data(image_path, mask_path):
    """Load in MRI slice and mask slice for train or validation datasets given their absolute paths

    Inputs: image_path: absolute path of the .bmp file where the MRI slice is stored
            mask_path: absolute path of the .bmp file where the mask is stored

    Outputs: image: tf.float64 normalized to [0, 1] of the MRI slice, size (xy_dim, xy_dim)
             mask_3d: tf.float64 of 0s and 1s of the mask, size (xy_dim, xy_dim, 2)
    """

    # Define vectors of bmp slices
    numbers = tf.range(start=1, limit=57, dtype=tf.int32)
    slice_nums = tf.strings.as_string(numbers, width=4, fill='0')

    # Remove .txt extension
    image_parts = tf.strings.split(image_path, sep='.')[0]
    mask_parts = tf.strings.split(mask_path, sep='.')[0]

    # Combine into list of strings and load in
    images = tf.strings.join([image_parts, "-", slice_nums, '.bmp'])
    masks = tf.strings.join([mask_parts, "-", slice_nums, '.bmp'])

    # tf.print(images)

    image = tf.map_fn(read_file, images, fn_output_signature=tf.uint8)
    mask = tf.map_fn(read_file, masks, fn_output_signature=tf.uint8)

    # tf.print(image.shape)

    image = tf.reshape(image, [224, 128, 56, 1])
    mask = tf.reshape(mask, [224, 128, 56, 1])

    mask_4d = assemble_4d_mask(mask)

    image = tf.cast(image, tf.float64) / 255.0
    mask_4d = tf.cast(mask_4d, tf.float64)

    return image, mask_4d


def create_dataset(image_dir, mask_dir, dataset_type):
    """Creates a tf.data.Dataset object with filenames listed

    Inputs: image_dir: absolute path of mri images with wildcard for all .bmp files (e.g. "R:/.../*.bmp")
            mask_dir: absolute path of mask images with wildcard for all .bmp files (e.g. "R:/.../*.bmp")
            dataset_type: one of ["train", "test", "val", "validation"] corresponding to the dataset type we're creating

    Outputs: dataset: tf.data.Dataset object containing filenames and mapping function for calling
    """

    # Create dataset from list of image files
    mri = tf.data.Dataset.list_files(image_dir, shuffle=False)
    mask = tf.data.Dataset.list_files(mask_dir, shuffle=False)

    dataset = tf.data.Dataset.zip((mri, mask))

    if dataset_type == 'train':
        dataset = dataset.map(load_data)
    elif dataset_type == 'validation' or dataset_type == 'val':
        dataset = dataset.map(load_data)
    elif dataset_type == 'test':
        dataset = dataset.map(load_test_data)
    return dataset


def get_dataset(batch_size, dataset_type, dataset):
    """Returns a tf.data.Dataset object given batch_size, dataset_type, and dataset selection

    Inputs: batch_size: (int), batch size for dataset
            dataset_type: one of ["train", "test", "val", "validation"] corresponding to the dataset type we're creating
            dataset: dataset flag from argparse calling, currently one of ["H", "HT"]

    Outputs: dataset: tf.data.Dataset object containing filenames and mapping function for calling
    """

    data_path = get_data_path(dataset)

    if dataset_type == 'train':
        images_dir = os.path.join(data_path, "train", "mri")
        mask_dir = os.path.join(data_path, "train", "mask")
    elif dataset_type == 'test':
        images_dir = os.path.join(data_path, "test", "mri")
        mask_dir = os.path.join(data_path, "test", "mask")
    elif dataset_type == 'val' or dataset_type == 'validation':
        images_dir = os.path.join(data_path, "val", "mri")
        mask_dir = os.path.join(data_path, "val", "mask")
    else:
        raise(ValueError, f"The value {dataset_type} for the variable dataset_type is not one of 'train', 'test', "
                          f"or 'val'")

    # Add bmp wildcard
    images_dir = os.path.join(images_dir, "*.txt")
    mask_dir = os.path.join(mask_dir, "*.txt")

    # Create dataset
    dataset = create_dataset(images_dir, mask_dir, dataset_type)

    # randomly shuffle if training
    if dataset_type == "train":
        dataset = dataset.shuffle(buffer_size=min([tf.data.experimental.cardinality(dataset).numpy() // 2, 4000]), seed=42)

    # Parallelize Data Loading Step
    # dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE)

    # Batch each dataset
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch batch into memory at a given time
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 4
    dataset = get_dataset(batch_size=batch_size, dataset_type='test', dataset="CHT")
    iterable = iter(dataset)
    out = next(iterable)

    print(out)
