import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from get_data_path import get_data_path
import matplotlib.pyplot as plt
from scipy import ndimage
from glob import glob
from PIL import Image


def isleftknee(subj):
    """Returns true if scan is of a left knee, so we can flip about the z dim"""
    left_scans = ["AS_042", "AS_043", "AS_044", "AS_045", "AS_046", "AS_047", "AS_098", "AS_099",
                  "AS_102", "AS_103", "AS_106", "AS_107", "AS_110", "AS_111", "AS_114", "AS_115",
                  "AS_118", "AS_119", "AS_122", "AS_123", "AS_126", "AS_127"]
    if subj in left_scans:
        return True
    else:
        return False


def assemble_4d_mask(mask_3d, tissue):
    """Assembles a 4D tf.Tensor of 1s and 0s pertaining to the patella and patellar cartilage

    Inputs: mask_3d (ndarray) of size (xy_dim, xy_dim, depth) containing 0, 1, and 2 pertaining to background, patella, and patellar cartilage, respectively

    Outputs: mask_4d (tf.tensor) of size (xy_dim, xy_dim, depth, 2) of 1s and 0s corresponding to P and PC
    """
    if tissue == 'p':
        p_mask_inds = np.equal(mask_3d, 1)
        mask = np.where(p_mask_inds, 1, 0)
    elif tissue == 'c':
        pc_mask_inds = np.equal(mask_3d, 2)
        mask = np.where(pc_mask_inds, 1, 0)
    else:
        raise ValueError("Invalid tissue type. Choose 'p' for patella or 'c' for patellar cartilage.")

    # Add 4th dim
    mask = np.expand_dims(mask, axis=-1)

    return mask


def load_images(dataset_name, dataset_type):
    data_path = get_data_path(dataset_name)

    mri_path = os.path.join(data_path, dataset_type, "mri")
    mask_path = os.path.join(data_path, dataset_type, "mask")

    subjs = os.listdir(mri_path)
    mris = np.zeros((len(subjs), 224, 128, 56))
    masks = np.zeros((len(subjs), 224, 128, 56))

    for i, subj in enumerate(subjs):
        print(f"Loading in {subj}")

        mri_files = sorted(glob(os.path.join(mri_path, subj, '*.bmp')))
        mask_files = sorted(glob(os.path.join(mask_path, subj, '*.bmp')))

        mri = np.zeros((224, 128, len(mri_files)))
        mask = np.zeros((224, 128, len(mask_files)))

        for j, mri_file in enumerate(mri_files):
            mri[:, :, j] = np.array(Image.open(mri_file))
            mask[:, :, j] = np.array(Image.open(mask_files[j]))

        if isleftknee(subj):
            mri = np.flip(mri, axis=-1)
            mask = np.flip(mask, axis=-1)

        mris[i, :, :, :] = mri
        masks[i, :, :, :] = mask

    return mris, masks


def get_dataset(dataset_name, dataset_type, batch_size, tissue):
    data_path = get_data_path(dataset_name)
    mri_path = os.path.join(data_path, dataset_type, "mri")
    subjIDs = [folder for folder in os.listdir(mri_path)]

    mris, masks = load_images(dataset_name, dataset_type)
    masks = masks.astype(np.uint8)

    mri_3d = mris.astype(np.float32) / 255.0
    mri_3d = np.expand_dims(mri_3d, axis=-1)

    mask_4d = assemble_4d_mask(masks, tissue)
    mask_4d = mask_4d.astype(np.float32)

    dataset_mri = tf.data.Dataset.from_tensor_slices(tf.constant(mri_3d, dtype=tf.float32))
    dataset_mask = tf.data.Dataset.from_tensor_slices(tf.constant(mask_4d, dtype=tf.float32))
    dataset_subj = tf.data.Dataset.from_tensor_slices(subjIDs)

    if dataset_type == "train":
        dataset = tf.data.Dataset.zip((dataset_mri, dataset_mask))
        dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    elif dataset_type == "test":
        dataset = tf.data.Dataset.zip((dataset_subj, dataset_mri, dataset_mask))
        dataset = dataset.batch(batch_size=1)
    elif dataset_type == "val":
        dataset = tf.data.Dataset.zip((dataset_mri, dataset_mask))
        dataset = dataset.batch(batch_size=1)
    return dataset


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


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 4
    dataset = get_dataset(dataset_name="cHTCO", dataset_type="val", batch_size=batch_size, tissue='p')

    i = iter(dataset)
    out = next(i)
    mri, mask = out
    # subj = subj.numpy()
    mri = mri.numpy()
    mask = mask.numpy()
    print()
