import numpy as np
import scipy
from get_data_path import get_data_path
import os
from PIL import Image
import random
import matplotlib.pyplot as plt
import shutil
import pickle


def get_img_names(train_path_labels):
    files = os.listdir(train_path_labels)
    files.sort()
    return files


def get_subj_names(image_names):
    sid = []
    for image_name in image_names:
        sid.append(image_name.split("-")[0])

    # Get unique values and sort
    sid = list(set(sid))
    sid.sort()
    return sid


def get_image_names_for_subj(subj, train_path_labels, train_path_imgs, image_names):
    subj_image_names = [image_name for image_name in image_names if subj in image_name]
    label_names = [train_path_labels + "/" + subj_image for subj_image in subj_image_names]
    mri_names = [train_path_imgs + "/" + subj_image for subj_image in subj_image_names]
    return label_names, mri_names


def assemble_mri_volume(image_names, xy_dim=256):
    volume = np.zeros((xy_dim, xy_dim, 120))
    for slice, image_name in enumerate(image_names):
        image_here = Image.open(image_name)
        image_np = np.asarray(image_here)
        volume[:, :, slice] = image_np
    volume = volume.astype(np.float64) / 255.0
    return volume


def assemble_mask_volume(image_names, xy_dim=256):
    pat_vol = np.zeros((xy_dim, xy_dim, 120))
    pat_cart_vol = np.zeros((xy_dim, xy_dim, 120))
    for slice, image_name in enumerate(image_names):
        image_here = Image.open(image_name)
        image_np = np.asarray(image_here)

        pat_here = np.where(image_np == 1, 1, 0)
        pat_cart_here = np.where(image_np == 2, 1, 0)

        pat_vol[:, :, slice] = pat_here
        pat_cart_vol[:, :, slice] = pat_cart_here

    return pat_vol, pat_cart_vol


def get_random_motion(voxel_lengths, mean_val):
    sig12 = np.random.poisson(mean_val*100, 2)/100  # in x and y directions
    decrease_in_z = voxel_lengths[1] / voxel_lengths[2]  # decreasing in motion artifact due to increase in length scale in the z direction
    sig3 = np.random.poisson(100*mean_val * decrease_in_z, 1)/100
    sigma = (sig12[0], sig12[1], sig3[0])
    return sigma


def save_augmented_slices(mri_blur, subj, mri_names, label_names, idx):
    mri_blur = mri_blur * 255
    mri_blur = mri_blur.astype(np.uint8)

    for slice_num, (mri_name, label_name) in enumerate(zip(mri_names, label_names)):
        slice_str = four_digit_number(slice_num+1)

        # copy the mask
        mask_source = label_name
        mask_dest = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Augmentations/sig" + str(idx) + "/train/mask/a" + subj + "-" + slice_str + ".bmp"
        shutil.copy(mask_source, mask_dest)

        # Write the current slice of the MRI image
        current_mri = mri_blur[:, :, slice_num]
        mri_img = Image.fromarray(current_mri)
        mri_dest = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Augmentations/sig" + str(idx) + "/train/mri/a" + subj + "-" + slice_str + ".bmp"
        mri_img.save(mri_dest)

    return


def four_digit_number(int_num):
    if int_num < 10:
        return "000" + str(int_num)
    elif int_num < 100:
        return "00" + str(int_num)
    elif int_num < 1000:
        return "0" + str(int_num)
    else:
        raise (ValueError, f"Number {int_num} is not a valid slice number")


def store_sigma(sigma_dict, sigma_list, idx, mean_val):
    sigma_dict[idx] = {"mean_value": mean_val, "sigmas": sigma_list}
    return sigma_dict


def save_sigma(sigma_dict):
    with open("./augmentation/parameters.pkl", "wb") as f:
        pickle.dump(sigma_dict, f)


if __name__ == "__main__":
    random.seed(42)
    data_path = get_data_path()

    train_path_labels = data_path + "/train" + "/mask"
    train_path_imgs = data_path + "/train" + "/mri"

    img_names = get_img_names(train_path_labels)
    sid = get_subj_names(img_names)

    voxel_lengths = [0.3, 0.3, 1.0]  # voxel lengths in mm
    mean_vals = [0.25, 0.5, 0.75, 1, 1.25]  # mean vals in which to save augmented data
    sigma_dict = {}
    sigma_list = []
    for idx, mean_val in enumerate(mean_vals):

        for subj in sid:
            label_names, mri_names = get_image_names_for_subj(subj, train_path_labels, train_path_imgs, img_names)

            pat_vol, pat_cart_vol = assemble_mask_volume(label_names)
            mri_vol = assemble_mri_volume(mri_names)

            sigma = get_random_motion(voxel_lengths, mean_val=mean_val)
            sigma_list = [sigma_list, [list(sigma)]]

            mri_blur = scipy.ndimage.gaussian_filter(mri_vol, sigma=sigma, order=0)

            save_augmented_slices(mri_blur, subj, mri_names, label_names, idx)

            print(f"Subject {subj} done")

        store_sigma(sigma_dict, sigma_list, idx, mean_val)

    save_sigma(sigma_dict)
    print(f"Mean Value {mean_vals} done")