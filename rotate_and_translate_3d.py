import numpy as np
from PIL import Image
import os
from get_data_path import get_data_path
import argparse
import scipy
import matplotlib.pyplot as plt
import shutil
import glob

parser = argparse.ArgumentParser(description="Augmentation Options")
parser.add_argument("-d", "--dataset", default="cHTCO-Group", help="Recent dataset is cHT")
parser.add_argument("-n", "--naug", default=5, help="Number of Augmentations to Make", type=int)
args = parser.parse_args()

print(f"Input Dataset: {args.dataset}")
print(f"Number of Augmentations: {args.naug}")


def rotate_images(mri, mask, deg):
    rot_mri = scipy.ndimage.rotate(mri, deg, reshape=False, order=3, mode='constant', cval=0, prefilter=False)
    rot_mask = scipy.ndimage.rotate(mask, deg, reshape=False, order=0, mode='constant', cval=0, prefilter=False)
    return rot_mri, rot_mask


def translate_images(mri, mask, trans):
    trans_mri = scipy.ndimage.shift(mri, shift=(trans[0], trans[1]), mode='constant', cval=0)
    trans_mask = scipy.ndimage.shift(mask, shift=(trans[0], trans[1]), mode='constant', cval=0)
    return trans_mri, trans_mask


def save_images(scan, trans_mri, trans_mask, save_data_path, new_file_name):
    # Save file full paths
    mri_file_path = os.path.join(save_data_path, "train", "mri", scan, new_file_name)
    mask_file_path = os.path.join(save_data_path, "train", "mask", scan, new_file_name)

    # If float, convert to uint8
    if trans_mask.dtype == np.float64:
        trans_mask = trans_mask.astype(np.uint8)
        trans_mri = trans_mri.astype(np.uint8)

    # Save images
    mri_img = Image.fromarray(trans_mri)
    mask_img = Image.fromarray(trans_mask)

    mri_img.save(mri_file_path)
    mask_img.save(mask_file_path)
    return


def set_up_dataset_directory(save_data_path):
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
        os.mkdir(os.path.join(save_data_path, "test"))
        os.mkdir(os.path.join(save_data_path, "train"))
        os.mkdir(os.path.join(save_data_path, "val"))

        # os.mkdir(os.path.join(save_data_path, "test", "mri"))
        os.mkdir(os.path.join(save_data_path, "train", "mri"))
        # os.mkdir(os.path.join(save_data_path, "val", "mri"))

        # os.mkdir(os.path.join(save_data_path, "test", "mask"))
        os.mkdir(os.path.join(save_data_path, "train", "mask"))
        # os.mkdir(os.path.join(save_data_path, "val", "mask"))


def rot_and_trans_bounds(a):
    """Returns the bounds for rotation and translation based on slurm job task array"""
    rot_trans_dict = [{"rot": 10.0, "trans": 5.0},
                      {"rot": 10.0, "trans": 10.0},
                      {"rot": 10.0, "trans": 20.0},
                      {"rot": 10.0, "trans": 30.0},
                      {"rot": 20.0, "trans": 5.0},
                      {"rot": 20.0, "trans": 10.0},
                      {"rot": 20.0, "trans": 20.0},
                      {"rot": 20.0, "trans": 30.0},
                      {"rot": 30.0, "trans": 5.0},
                      {"rot": 30.0, "trans": 10.0},
                      {"rot": 30.0, "trans": 20.0},
                      {"rot": 30.0, "trans": 30.0},
                      {"rot": 40.0, "trans": 5.0},
                      {"rot": 40.0, "trans": 10.0},
                      {"rot": 40.0, "trans": 20.0},
                      {"rot": 40.0, "trans": 30.0}]

    combo = rot_trans_dict[a]
    rot = combo["rot"]
    tran = combo["trans"]
    return rot, tran


def get_slice_num(file):
    """Returns integer of 1 - 120 of slice number"""
    end_str = file.split("-")[-1]
    num_str = end_str.split(".")[0]
    return int(num_str)


def rotate_volumes(mri_vol, mask_vol, rot):
    """Rotates 3D volumes by a random degree in x, y, and z dimensions"""
    # Randomly rotate in x, y, and z dimensions
    x_rot = np.random.uniform(-rot, rot)
    y_rot = np.random.uniform(-rot, rot)
    z_rot = np.random.uniform(-rot, rot)

    # Rotate only about Z axis
    rot_mri = scipy.ndimage.rotate(mri_vol, x_rot, axes=(0, 1), reshape=False, order=3, mode='constant', cval=0, prefilter=False)
    rot_mask = scipy.ndimage.rotate(mask_vol, x_rot, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0, prefilter=False)

    # Rotate MRI and mask volumes about all 3 axes
    # rot_mri = scipy.ndimage.rotate(mri_vol, x_rot, axes=(0, 1), reshape=False, order=3, mode='constant', cval=0, prefilter=False)
    # rot_mri = scipy.ndimage.rotate(rot_mri, y_rot, axes=(0, 2), reshape=False, order=3, mode='constant', cval=0, prefilter=False)
    # rot_mri = scipy.ndimage.rotate(rot_mri, z_rot, axes=(1, 2), reshape=False, order=3, mode='constant', cval=0, prefilter=False)
    #
    # rot_mask = scipy.ndimage.rotate(mask_vol, x_rot, axes=(0, 1), reshape=False, order=0, mode='constant', cval=0, prefilter=False)
    # rot_mask = scipy.ndimage.rotate(rot_mask, y_rot, axes=(0, 2), reshape=False, order=0, mode='constant', cval=0, prefilter=False)
    # rot_mask = scipy.ndimage.rotate(rot_mask, z_rot, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0, prefilter=False)

    return rot_mri, rot_mask


if __name__ == "__main__":
    # print(f"args.arr: {args.arr}")

    # Get rotation and translation bounds from slurm job task array
    # rot, t = rot_and_trans_bounds(args.arr)

    rot = 15.0
    print(f"Rotation Bounds: +/- {rot} degrees")

    # Set random seed
    np.random.seed(42)

    # Load in image path
    data_path = get_data_path("cHTCO-Group")
    train_path = os.path.join(data_path, "train")

    # Save data path: make it if it doesn't exist
    save_data_path = get_data_path("cHTCO-Group5Z")

    print(f"Saving new images to {save_data_path}")

    set_up_dataset_directory(save_data_path)

    # move_test_and_val(data_path, save_data_path)

    # List images
    mri_path = os.path.join(train_path, "mri")
    mask_path = os.path.join(train_path, "mask")

    scans = os.listdir(mri_path)

    for scan in scans:
        # make mri and mask directories in destination
        if not os.path.exists(os.path.join(save_data_path, "train", "mri", scan)):
            os.mkdir(os.path.join(save_data_path, "train", "mri", scan))
            os.mkdir(os.path.join(save_data_path, "train", "mask", scan))

        mask_vol = np.zeros((224, 128, 56))
        mri_vol = np.zeros((224, 128, 56))

        for idx, file in enumerate(os.listdir(os.path.join(mri_path, scan))):
            slice_num = get_slice_num(file)

            mri_file = os.path.join(mri_path, scan, file)
            mask_file = os.path.join(mask_path, scan, file)

            mri_img = Image.open(mri_file)
            mri = np.asarray(mri_img)
            mask_img = Image.open(mask_file)
            mask = np.asarray(mask_img)

            mask_vol[:, :, idx] = mask
            mri_vol[:, :, idx] = mri

            # Save original image
            save_images(scan, mri, mask, save_data_path, file)

        print(f"Saved {scan}")

        # rotate volumes n number of times
        for i in range(args.naug):
            mri_rot, mask_rot = rotate_volumes(mri_vol, mask_vol, rot)

            # Create directories
            scan_aug = f"a{i}{scan}"
            if not os.path.exists(os.path.join(save_data_path, "train", "mri", scan_aug)):
                os.mkdir(os.path.join(save_data_path, "train", "mri", scan_aug))
                os.mkdir(os.path.join(save_data_path, "train", "mask", scan_aug))

            # iterate through slices and save
            for idx in range(mri_rot.shape[2]):
                filename = f"{scan_aug}-{idx+1:04}.bmp"
                save_images(scan_aug, mri_rot[:, :, idx], mask_rot[:, :, idx], save_data_path, filename)

            print(f"Saved {scan_aug}")
