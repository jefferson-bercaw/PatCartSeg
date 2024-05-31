# This script analyzes the data from Paranjape et al. 2020
import numpy as np
import os
import pickle
import tensorflow as tf
import itertools

from dice_loss_function import dice_loss
from get_data_path import get_data_path
from evaluate import load_model


def get_paranjape_dataset(image_dir, batch_size):
    """Returns a tf.data.Dataset object containing the filenames of the Paranjape dataset"""
    dataset = tf.data.Dataset.list_files(image_dir, shuffle=False)
    dataset = dataset.map(load_data)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def load_data(image_path):
    """Load in MRI slice and return the slice tensor and the filename"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image)

    filename = tf.strings.split(image_path, os.path.sep)[-1]

    return filename, image


def get_model_filename(model_name):
    """Assembles model absolute filename from model name"""
    model_filename = os.path.join("..", "models", model_name)
    return model_filename


def process_predicted_label(pred_label):
    """Takes in (batch_size, 256, 256, 2) ndarray, and returns two, (256, 256, 12) binary ndarrays of P and PC preds"""

    thresholded_label = (pred_label >= 0.5)
    binary_data = thresholded_label.astype(np.uint8)

    pat = np.squeeze(binary_data[:, :, :, 0])
    pat_cart = np.squeeze(binary_data[:, :, :, 1])

    # Transpose to (256, 256, batch_size)
    pat = np.transpose(pat, (1, 2, 0))
    pat_cart = np.transpose(pat_cart, (1, 2, 0))

    return pat, pat_cart


def parse_scan_name(filename):
    """Takes in tf.tensor of b-strings and returns the scan name (str). Also, checks that the first image is here"""
    filename_str_start = filename.numpy()[0].decode()

    # Remove extension, get slice number, assert it is 1, and return the scan name
    no_ext = filename_str_start.split(".")[0]
    num = int(no_ext.split("-")[1])
    assert num == 1, (f"Images are out of order! First image in this batch is not first in scan! "
                      f"First batch image: {filename_str_start}")

    scan_name = no_ext.split("-")[0]

    print(f"Starting scan {scan_name}")

    return scan_name


def save_volumes(pat_vol, pat_cart_vol, model_name, scan_name):
    start_path = get_data_path("Paranjape_Volumes")
    volume_path = os.path.join(start_path, model_name[0:-3])

    # If paths don't exist, create them
    if not os.path.exists(start_path):
        os.mkdir(start_path)
    if not os.path.exists(volume_path):
        os.mkdir(volume_path)

    # Assert shape is correct
    assert pat_vol.shape == (256, 256, 120), (f"Patella volume being saved is not (256, 256, 120)! "
                                              f"Shape {pat_vol.shape}")
    assert pat_cart_vol.shape == (256, 256, 120), (f"Patellar cartilage volume being saved is not (256, 256, 120)! "
                                                   f"Shape {pat_cart_vol.shape}")

    # Save scans
    np.savez(os.path.join(volume_path, scan_name), pat_vol=pat_vol, pat_cart_vol=pat_cart_vol)
    return


if __name__ == "__main__":
    # Options/Declarations:
    predict_volumes_option = True
    model_name = "unet_2024-05-29_17-21-09_cHT5.h5"
    batch_size = 12
    n_slices = 120
    batches_per_scan = n_slices // batch_size

    if predict_volumes_option:
        # Load in dataset
        image_dir = get_data_path("Paranjape_Cropped") + os.sep + "*.bmp"
        dataset = get_paranjape_dataset(image_dir, batch_size=batch_size)
        iterable = iter(dataset)
        iterable = itertools.islice(iterable, 310, None)

        # Load in model
        model_filename = get_model_filename(model_name)
        model = load_model(model_filename)

        # Iterate through scans in dataset, make predictions, create volumes, save ndarrays
        for i in range(32, len(dataset) // batches_per_scan):

            for cur_batch_num in range(1, batches_per_scan + 1):
                filename, mri = next(iterable)
                pred_label = model.predict(mri)
                pat, pat_cart = process_predicted_label(pred_label)

                # If we're predicting the first batch of images in the scan, assign volume to prediction,
                # and get scan name to save the volumes
                if cur_batch_num == 1:
                    pat_vol = pat
                    pat_cart_vol = pat_cart
                    scan_name = parse_scan_name(filename)

                # If we're not predicting first batch, append to current batch
                elif cur_batch_num <= batches_per_scan:
                    pat_vol = np.concatenate((pat_vol, pat), axis=2)
                    pat_cart_vol = np.concatenate((pat_cart_vol, pat_cart), axis=2)

                # If we've predicted final batch for scan, save volume
                if cur_batch_num == batches_per_scan:
                    save_volumes(pat_vol, pat_cart_vol, model_name, scan_name)
                    print(f"Saved {scan_name}, scan {i} of {(len(dataset) // batches_per_scan) - 1}")
