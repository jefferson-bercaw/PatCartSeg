import numpy as np
import os
from get_data_path import get_data_path
import scipy
from PIL import Image


def voting_masks(pat, pat_cart):
    """Returns the hard and soft masks of the patella and patellar cartilage using model voting"""
    # Soft voting
    pat_soft = np.mean(pat, axis=2)
    pat_cart_soft = np.mean(pat_cart, axis=2)

    pat_soft_mask = pat_soft >= 0.5
    pat_cart_soft_mask = pat_cart_soft >= 0.5

    # Hard voting
    pat_hard = pat >= 0.5
    pat_cart_hard = pat_cart >= 0.5

    pat_hard = scipy.stats.mode(pat_hard, axis=2)
    pat_cart_hard = scipy.stats.mode(pat_cart_hard, axis=2)

    # Squeeze out last dimension
    pat_hard_mask = np.squeeze(pat_hard)
    pat_cart_hard_mask = np.squeeze(pat_cart_hard)

    # Convert all to uint8
    pat_soft_mask = pat_soft_mask.astype(np.uint8)
    pat_cart_soft_mask = pat_cart_soft_mask(np.uint8)
    pat_hard_mask = pat_hard_mask.astype(np.uint8)
    pat_cart_hard_mask = pat_cart_hard_mask.astype(np.uint8)

    return pat_hard_mask, pat_cart_hard_mask, pat_soft_mask, pat_cart_soft_mask


def prep_save_path():
    if not os.path.exists(os.path.join(os.getcwd(), "results", "model_voting_hard")):
        os.mkdir(os.path.join(os.getcwd(), "results", "model_voting_hard"))
        os.mkdir(os.path.join(os.getcwd(), "results", "model_voting_hard", "pat"))
        os.mkdir(os.path.join(os.getcwd(), "results", "model_voting_hard", "pat_cart"))
    if not os.path.exists(os.path.join(os.getcwd(), "results", "model_voting_soft")):
        os.mkdir(os.path.join(os.getcwd(), "results", "model_voting_soft"))
        os.mkdir(os.path.join(os.getcwd(), "results", "model_voting_soft", "pat"))
        os.mkdir(os.path.join(os.getcwd(), "results", "model_voting_soft", "pat_cart"))


def save_result(file, pat_hard, pat_cart_hard, pat_soft, pat_cart_soft):
    img_p_hard = Image.fromarray(pat_hard)
    img_pc_hard = Image.fromarray(pat_cart_hard)
    img_p_soft = Image.fromarray(pat_soft)
    img_pc_soft = Image.fromarray(pat_cart_soft)

    filename = file.split(".")[0] + ".bmp"

    img_p_hard.save(os.path.join(os.getcwd(), "results", "model_voting_hard", "pat", filename))
    img_pc_hard.save(os.path.join(os.getcwd(), "results", "model_voting_hard", "pat_cart", filename))
    img_p_soft.save(os.path.join(os.getcwd(), "results", "model_voting_soft", "pat", filename))
    img_pc_soft.save(os.path.join(os.getcwd(), "results", "model_voting_soft", "pat_cart", filename))


if __name__ == "__main__":
    date_times = ["unet_2024-05-22_00-17-51_cHT5", "unet_2024-05-22_03-06-08_cHT5", "unet_2024-05-21_13-59-19_cHT5",
                  "unet_2024-05-21_20-50-23_cHT5", "unet_2024-05-22_07-43-29_cHT5", "unet_2024-05-21_15-36-23_cHT5",
                  "unet_2024-05-22_06-41-03_cHT5", "unet_2024-05-22_04-18-45_cHT5", "unet_2024-05-21_23-18-29_cHT5"]

    files = os.listdir(os.path.join(os.getcwd(), "results", date_times[0], "pat_prob"))
    prep_save_path()

    for file in files:
        # Initialize probability maps
        pat = np.zeros((256, 256, 9))
        pat_cart = np.zeros((256, 256, 9))
        for idx, date_time in date_times:
            # Filename
            pat_filepath = os.path.join(os.getcwd(), "results", date_time, "pat_prob", file)
            pat_cart_filepath = os.path.join(os.getcwd(), "results", date_time, "pat_cart_prob", file)

            # Numpy Arrays
            pat[:, :, idx] = np.load(pat_filepath)
            pat_cart[:, :, idx] = np.load(pat_cart_filepath)

        # Get predicted masks based on hard and soft voting
        pat_hard, pat_cart_hard, pat_soft, pat_cart_soft = voting_masks(pat, pat_cart)

        # Save result
        save_result(file, pat_hard, pat_cart_hard, pat_soft, pat_cart_soft)
