import numpy as np
from PIL import Image
import os
from get_data_path import get_data_path
import argparse
import scipy
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Training Options")
parser.add_argument("-d", "--dataset", help="Dataset is either H or HT")
parser.add_argument("-n", "--naug", help="Number of Augmentations to Make", type=int)
args = parser.parse_args(["-d", "HT", "-n", "5"])
print(args.dataset)
print(args.naug)


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


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Load in image path
    data_path = get_data_path(args.dataset)
    train_path = os.path.join(data_path, "train")

    # Save data path: make it if it doesn't exist
    save_data_path = get_data_path(f"{args.dataset}{args.naug}")
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

    # List images
    mri_path = os.path.join(train_path, "mri")
    mask_path = os.path.join(train_path, "mask")

    files = os.listdir(mri_path)

    for file in files:
        mri_file = os.path.join(mri_path, file)
        mask_file = os.path.join(mask_path, file)

        mri_img = Image.open(mri_file)
        mri = np.asarray(mri_img)

        mask_img = Image.open(mask_file)
        mask = np.asarray(mask_img)

        # Save original image
        save_images(mri, mask, save_data_path, file)

        # Generate random rotations/translations
        degs = np.random.uniform(-30.0, 30.0, size=args.naug)
        trans = np.random.randint(-20, 21, size=(args.naug, 2))

        for idx, deg in enumerate(degs):
            tran = trans[idx, :]

            # Rotate
            rot_mri, rot_mask = rotate_images(mri, mask, deg)

            # Translate
            trans_mri, trans_mask = translate_images(rot_mri, rot_mask, tran)

            # Save
            save_images(trans_mri, trans_mask, save_data_path, f"a{idx}{file}")

            # Plot
            plt.imshow(trans_mri)
            plt.show()

            plt.imshow(trans_mask)
            plt.show()

            plt.imshow(mri)
            plt.show()

            plt.imshow(mask)
            plt.show()

            print(f"Tran: {tran}")
            print(f"Deg: {deg}")

    print("Pause")