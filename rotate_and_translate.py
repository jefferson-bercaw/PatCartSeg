import numpy as np
from PIL import Image
import os
from get_data_path import get_data_path
import argparse
import scipy
import matplotlib.pyplot as plt
import shutil

parser = argparse.ArgumentParser(description="Augmentation Options")
parser.add_argument("-d", "--dataset", help="Recent dataset is cHT")
parser.add_argument("-n", "--naug", help="Number of Augmentations to Make", type=int)
parser.add_argument("-a", "--arr", help="Slurm Task Array Number Controlling Rotation and Translation", type=int)
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


def save_images(trans_mri, trans_mask, save_data_path, new_file_name):
    # Save file full paths
    mri_file_path = os.path.join(save_data_path, "train", "mri", new_file_name)
    mask_file_path = os.path.join(save_data_path, "train", "mask", new_file_name)

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

        os.mkdir(os.path.join(save_data_path, "test", "mri"))
        os.mkdir(os.path.join(save_data_path, "train", "mri"))
        os.mkdir(os.path.join(save_data_path, "val", "mri"))

        os.mkdir(os.path.join(save_data_path, "test", "mask"))
        os.mkdir(os.path.join(save_data_path, "train", "mask"))
        os.mkdir(os.path.join(save_data_path, "val", "mask"))


def move_test_and_val(data_path, save_data_path):
    """Moving all contents from the data_path to the save_data_path"""
    val_src = os.path.join(data_path, "val")
    test_src = os.path.join(data_path, "test")

    val_dest = os.path.join(save_data_path, "val")
    test_dest = os.path.join(save_data_path, "test")

    # Remove destination directory if it already exists
    if os.path.exists(val_dest):
        shutil.rmtree(val_dest)
    if os.path.exists(test_dest):
        shutil.rmtree(test_dest)

    # Copy
    shutil.copytree(val_src, val_dest)
    shutil.copytree(test_src, test_dest)
    return


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


if __name__ == "__main__":
    # print(f"args.arr: {args.arr}")

    # Get rotation and translation bounds from slurm job task array
    # rot, t = rot_and_trans_bounds(args.arr)

    rot = 20.0
    t = 20.0

    print(f"Rotation Bounds: +/- {rot} degrees")
    print(f"Translation Bounds: +/- {t} px")

    # Set random seed
    np.random.seed(42)

    # Load in image path
    data_path = get_data_path("cHT")

    train_path = os.path.join(data_path, "train")

    # Save data path: make it if it doesn't exist
    save_data_path = get_data_path("cHT5")

    print(f"Saving new images to {save_data_path}")

    set_up_dataset_directory(save_data_path)

    move_test_and_val(data_path, save_data_path)

    # List images
    mri_path = os.path.join(train_path, "mri")
    mask_path = os.path.join(train_path, "mask")

    files = os.listdir(mri_path)

    for file_num, file in enumerate(files):
        mri_file = os.path.join(mri_path, file)
        mask_file = os.path.join(mask_path, file)

        mri_img = Image.open(mri_file)
        mri = np.asarray(mri_img)

        mask_img = Image.open(mask_file)
        mask = np.asarray(mask_img)

        # Save original image
        save_images(mri, mask, save_data_path, file)

        # Generate random rotations/translations
        degs = np.random.uniform(-rot, rot, size=5)
        trans = np.random.randint(int(-1 * t), int(t+1), size=(5, 2))

        for idx, deg in enumerate(degs):
            tran = trans[idx, :]

            # Rotate
            rot_mri, rot_mask = rotate_images(mri, mask, deg)

            # Translate
            trans_mri, trans_mask = translate_images(rot_mri, rot_mask, tran)

            # Save
            save_images(trans_mri, trans_mask, save_data_path, f"a{idx}{file}")

        if file_num % (50 * (5+1)):
            print(f"File {file_num * (5+1)} of {len(files) * (5+1)}")
