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


def move_test_and_val(data_path, save_data_path, to_exclude):
    """Moving all contents from the data_path to the save_data_path"""
    val_src = os.path.join(data_path, "val")
    test_src = os.path.join(data_path, "test")

    val_dest = os.path.join(save_data_path, "val")
    test_dest = os.path.join(save_data_path, "test")

    # Get source and destination paths for val and test dataset for masks and mris
    val_mask_src = os.path.join(val_src, "mask")
    val_mri_src = os.path.join(val_src, "mri")
    val_mask_dest = os.path.join(val_dest, "mask")
    val_mri_dest = os.path.join(val_dest, "mri")

    test_mask_src = os.path.join(test_src, "mask")
    test_mri_src = os.path.join(test_src, "mri")
    test_mask_dest = os.path.join(test_dest, "mask")
    test_mri_dest = os.path.join(test_dest, "mri")

    srcs = [val_mask_src, val_mri_src, test_mask_src, test_mri_src]
    dests = [val_mask_dest, val_mri_dest, test_mask_dest, test_mri_dest]

    for src, dest in zip(srcs, dests):
        files = os.listdir(src)
        for file in files:
            slice_num = get_slice_num(file)
            if (slice_num > to_exclude // 2) and (slice_num <= 120 - to_exclude // 2):
                source_path = os.path.join(src, file)
                dest_path = os.path.join(dest, file)
                shutil.copyfile(source_path, dest_path)
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


def get_slice_num(file):
    """Returns integer of 1 - 120 of slice number"""
    end_str = file.split("-")[-1]
    num_str = end_str.split(".")[0]
    return int(num_str)


def remove_outer_bounds(to_exclude=50):
    """Removes training, testing, validation images outside to_exclude//2 bounds from given dataset"""

    data_path = get_data_path("cHT")
    
    train_path = os.path.join(data_path, "train")

    # Save data path: make it if it doesn't exist
    save_data_path = get_data_path("ctHT")

    print(f"Saving new images to {save_data_path}")

    set_up_dataset_directory(save_data_path)

    move_test_and_val(data_path, save_data_path, to_exclude)

    # List images
    mri_path = os.path.join(train_path, "mri")
    mask_path = os.path.join(train_path, "mask")

    files = os.listdir(mri_path)

    for file_num, file in enumerate(files):
        slice_num = get_slice_num(file)

        if (slice_num > to_exclude // 2) and (slice_num <= 120 - to_exclude // 2):

            mri_file = os.path.join(mri_path, file)
            mask_file = os.path.join(mask_path, file)

            mri_img = Image.open(mri_file)
            mri = np.asarray(mri_img)

            mask_img = Image.open(mask_file)
            mask = np.asarray(mask_img)

            # Save original image
            save_images(mri, mask, save_data_path, file)
            if file_num % 100 == 0:
                print(f"File {file_num} of {len(files)}")

    return


if __name__ == "__main__":
    # print(f"args.arr: {args.arr}")

    # Get rotation and translation bounds from slurm job task array
    # rot, t = rot_and_trans_bounds(args.arr)
    rot = 20.0
    t = 20.0
    to_exclude = 50  # excludes n images (n/2 on either side)
    print(f"Rotation Bounds: +/- {rot} degrees")
    print(f"Translation Bounds: +/- {t} px")

    # Set random seed
    np.random.seed(42)

    # Load in image path
    # data_path = get_data_path(args.dataset)
    data_path = get_data_path("cHT")
    train_path = os.path.join(data_path, "train")

    # Save data path: make it if it doesn't exist
    save_data_path = get_data_path("ctHT5")

    print(f"Saving new images to {save_data_path}")

    set_up_dataset_directory(save_data_path)

    move_test_and_val(data_path, save_data_path, to_exclude)

    # List images
    mri_path = os.path.join(train_path, "mri")
    mask_path = os.path.join(train_path, "mask")

    files = os.listdir(mri_path)

    for file_num, file in enumerate(files):
        slice_num = get_slice_num(file)

        if (slice_num > to_exclude // 2) and (slice_num <= 120 - to_exclude // 2):

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

            if file_num % (50 * (6)):
                print(f"File {file_num * (6)} of {len(files) * 6}")
