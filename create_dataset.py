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
    mask = tf.squeeze(mask, axis=-3)
    # tf.print(mask.shape)

    return mask


def load_train_scan(mri_folder, mask_folder):

    # train_path = get_data_path("CHT-Group")
    # mri_path = os.path.join(train_path, "train", "mri", scan_folder)
    # mask_path = os.path.join(train_path, "train", "mask", scan_folder)

    # tf.print(mri_path)

    mri_files = sorted(glob(os.path.join(mri_folder.numpy().decode("utf-8"), "*.bmp")))
    mask_files = sorted(glob(os.path.join(mask_folder.numpy().decode("utf-8"), "*.bmp")))

    # Load each image, resize if needed, and stack them
    mris = [tf.image.decode_bmp(tf.io.read_file(img_file)) for img_file in mri_files]
    masks = [tf.image.decode_bmp(tf.io.read_file(img_file)) for img_file in mask_files]

    # Optionally resize images to (128, 224)
    mris = [tf.image.resize(img, [224, 128]) for img in mris]
    masks = [tf.image.resize(img, [224, 128]) for img in masks]

    # Stack images to create a (224, 128, 56) tensor
    mri_3d = tf.stack(mris, axis=-1)
    mask_3d = tf.stack(masks, axis=-1)

    mri_3d = tf.transpose(mri_3d, perm=[0, 1, 3, 2])
    mask_4d = assemble_4d_mask(mask_3d)

    mri_3d = tf.cast(mri_3d, tf.float64) / 255.0
    mask_4d = tf.cast(mask_4d, tf.float64)

    return mri_3d, mask_4d


def load_test_scan(mri_folder, mask_folder):

    # train_path = get_data_path("CHT-Group")
    # mri_path = os.path.join(train_path, "train", "mri", scan_folder)
    # mask_path = os.path.join(train_path, "train", "mask", scan_folder)

    # tf.print(mri_path)

    mri_files = sorted(glob(os.path.join(mri_folder.numpy().decode("utf-8"), "*.bmp")))
    mask_files = sorted(glob(os.path.join(mask_folder.numpy().decode("utf-8"), "*.bmp")))

    # Load each image, resize if needed, and stack them
    mris = [tf.image.decode_bmp(tf.io.read_file(img_file)) for img_file in mri_files]
    masks = [tf.image.decode_bmp(tf.io.read_file(img_file)) for img_file in mask_files]

    # Optionally resize images to (128, 224)
    mris = [tf.image.resize(img, [224, 128]) for img in mris]
    masks = [tf.image.resize(img, [224, 128]) for img in masks]

    # Stack images to create a (224, 128, 56) tensor
    mri_3d = tf.stack(mris, axis=-1)
    mask_3d = tf.stack(masks, axis=-1)

    mri_3d = tf.transpose(mri_3d, perm=[0, 1, 3, 2])
    mask_4d = assemble_4d_mask(mask_3d)

    mri_3d = tf.cast(mri_3d, tf.float64) / 255.0
    mask_4d = tf.cast(mask_4d, tf.float64)

    return mask_folder, mri_3d, mask_4d


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

    # Map the load_train_scan function to each pair of folders
    if dataset_type == 'train' or dataset_type == "val":
        dataset = dataset.map(lambda mri_folder, mask_folder: tf.py_function(
            func=load_train_scan,
            inp=[mri_folder, mask_folder],
            Tout=[tf.float64, tf.float64]))
    elif dataset_type == "test":
        dataset = dataset.map(lambda mri_folder, mask_folder: tf.py_function(
            func=load_test_scan,
            inp=[mri_folder, mask_folder],
            Tout=[tf.string, tf.float64, tf.float64]))

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
    dataset = get_dataset(batch_size=batch_size, dataset_type='train', dataset="CHT-Group")
    iterable = iter(dataset)
    out = next(iterable)
    mri, label = out

    print(f"MRI size: {mri.numpy().shape}")
    print(f"Mask size: {label.numpy().shape}")
