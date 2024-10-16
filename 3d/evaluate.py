import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import argparse
import datetime
import sys
import pandas as pd

# Add the main directory to the system path
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

from create_dataset import get_dataset
from dice_loss_function import dice_loss
from PIL import Image
from get_data_path import get_data_path
from multiclass_segment import save_model_info
from augmentation import assemble_mask_volume, assemble_mri_volume, four_digit_number


parser = argparse.ArgumentParser(description="Training Options")
parser.add_argument("--tissue", type=str, default='p', help="Tissue type to segment. Choose 'p' for patella or 'c' for patellar cartilage.")


def get_date_and_hour():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    return date_time


def get_history_filename(date_time):
    # Add extension
    date_time = date_time + ".pkl"
    history_filename = os.path.abspath(os.path.join('../history', date_time))
    return history_filename


def get_model_filename(date_time):
    date_time = date_time + ".h5"
    model_filename = os.path.abspath(os.path.join('../models', date_time))
    return model_filename


def load_history(history_filename):
    with open(history_filename, "rb") as f:
        history = pickle.load(f)
    return history


def load_model(model_filename):
    with tf.keras.utils.custom_object_scope({'dice_loss': dice_loss}):  # Register the custom loss function
        model = keras.models.load_model(model_filename)
    return model


def build_results_filename(model_filename):
    results_directory = model_filename[:-3]  # remove extension on model name

    # Change "model" to "results" in absolute path and join into a string
    sep = os.sep
    files = results_directory.split(sep)
    files[-2] = 'results'
    results_filename = sep.join(files)
    return results_filename


def get_hist_and_model(date_time):
    history_filename = get_history_filename(date_time)
    history = load_history(history_filename)
    model_filename = get_model_filename(date_time)
    model = load_model(model_filename)
    return history, model


def get_results_filename(date_time):
    model_filename = get_model_filename(date_time)
    results_filename = build_results_filename(model_filename)
    return results_filename


def prep_results_filepath(results_filename):
    if not os.path.exists(results_filename):
        os.mkdir(results_filename)

    examples_filename = os.path.join(results_filename, "examples")
    if not os.path.exists(examples_filename):
        os.mkdir(examples_filename)


def process_predicted_label(pred_label):
    # Predicted Masks
    thresholded_label = (pred_label >= 0.5)
    binary_data = thresholded_label.astype(np.uint8)

    # Binary predictions
    binary_predictions = np.squeeze(binary_data)

    # Probability masks
    probability_predictions = np.squeeze(pred_label)

    return binary_predictions, probability_predictions


def process_mri(mri):
    mri = np.squeeze(mri)
    return mri


def save_result(scan_name, date_time, pat, pat_prob, tissue):
    subj_str = scan_name.numpy()[0].decode()
    results_filename = get_results_filename(date_time)

    pat_filepath = os.path.join(results_filename, tissue)

    pat_prob_filepath = os.path.join(results_filename, tissue+"prob")

    # Make directories if they don't exist
    if not os.path.exists(pat_filepath):
        os.mkdir(pat_filepath)
    if not os.path.exists(pat_prob_filepath):
        os.mkdir(pat_prob_filepath)

    filename_npy = scan_name + ".npy"

    pat_filepath = os.path.join(pat_filepath, subj_str)
    pat_prob_filepath = os.path.join(pat_prob_filepath, subj_str)
    # print(f"Saving numpy arrays to {pat_prob_filepath} and {pat_cart_prob_filepath}")

    # Save the masks as BMP files
    for i in range(pat.shape[-1]):
        pat_img = Image.fromarray(pat[:, :, i] * 255)
        pat_img.save(pat_filepath + f"_{i:04d}.bmp")

    # Save the probability masks as NPY files
    np.save(pat_prob_filepath, pat_prob)
    return


def process_true_label(label):
    label = tf.squeeze(label, axis=0)
    pat = label[:, :, :, 0].numpy().astype(np.uint8)
    return pat


def count_positives(pred, true, positives):
    # Positives: [intersection, predicted, true]
    positives[0] += np.sum(np.where(pred + true == 2))
    positives[1] += np.sum(np.where(pred == 1))
    positives[2] += np.sum(np.where(true == 1))
    return positives


def calculate_dice(positives):
    return (2 * positives[0]) / (positives[1] + positives[2])


def save_metrics(date_time, metrics):
    results_filename = get_results_filename(date_time)
    print(f"Saving metrics to {os.path.join(results_filename, 'metrics.pkl')}")

    with open(os.path.join(results_filename, "metrics.pkl"), 'wb') as f:
        pickle.dump(metrics, f)
    return


def plot_mri_with_masks(mri_image, ground_truth_mask, predicted_mask, comp_filename, image_filename, tissue):
    # Define colors for ground truth and predicted masks
    gt_color = 'blue'
    pred_color = 'red'

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot MRI image
    ax.imshow(mri_image, cmap='gray')

    # Overlay ground truth mask
    alpha_gt = np.where(ground_truth_mask == 1, 0.4, 0)  # Set alpha based on mask values
    ax.imshow(ground_truth_mask, cmap='Blues', alpha=alpha_gt)

    # Overlay predicted mask
    alpha_pred = np.where(predicted_mask == 1, 0.4, 0)  # Set alpha based on mask values
    ax.imshow(predicted_mask, cmap='Reds', alpha=alpha_pred)

    # Create legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=gt_color, alpha=0.5, label='Ground Truth'),
        plt.Rectangle((0, 0), 1, 1, color=pred_color, alpha=0.5, label='Predicted')
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    # Saving figure
    image_filename = image_filename.numpy()[0].decode()
    plot_filename = os.path.join(comp_filename, tissue)

    if not os.path.exists(plot_filename):
        os.mkdir(plot_filename)

    plot_filename = os.path.join(plot_filename, image_filename)

    # Go to .png format
    plot_filename = plot_filename[:-3] + "png"
    plt.savefig(plot_filename, dpi=600)


def get_slice_list(p_truth_volume, pc_truth_volume):
    starting_slice = 0
    for i in range(70):
        slice_p_truth = p_truth_volume[:, :, i]
        slice_pc_truth = pc_truth_volume[:, :, i]

        p_sum = np.sum(slice_p_truth)
        pc_sum = np.sum(slice_pc_truth)

        if starting_slice == 0 and p_sum > 0 and pc_sum > 0:
            starting_slice = i

        if starting_slice > 0 and p_sum == 0 and pc_sum == 0:
            ending_slice = i - 1
            break

    slice_list = np.floor(np.linspace(starting_slice, ending_slice, 9))
    slice_list = slice_list.astype(int)
    return slice_list


def plot_mri_with_both_masks(subj_name, model_name):
    # Get Volumes for this subject and model prediction
    mri_volume, p_truth_volume, p_pred_volume, pc_truth_volume, pc_pred_volume = return_volumes(subj_name, model_name)

    p_pred_volume = np.where(p_pred_volume > 0, 1, 0)
    pc_pred_volume = np.where(pc_pred_volume > 0, 1, 0)

    opacity = 0.5
    alpha_p_truth = np.where(p_truth_volume == 1, opacity, 0)  # Set alpha based on mask values
    alpha_p_pred = np.where(p_pred_volume == 1, opacity, 0)  # Set alpha based on mask values
    alpha_pc_truth = np.where(pc_truth_volume == 1, opacity, 0)
    alpha_pc_pred = np.where(pc_pred_volume == 1, opacity, 0)

    # Remove extra slices to zoom into areas of interest
    mri_volume = mri_volume[80:220, :140, :]
    p_truth_volume = p_truth_volume[80:220, :140, :]
    pc_truth_volume = pc_truth_volume[80:220, :140, :]
    p_pred_volume = p_pred_volume[80:220, :140, :]
    pc_pred_volume = pc_pred_volume[80:220, :140, :]

    alpha_p_truth = alpha_p_truth[80:220, :140, :]
    alpha_p_pred = alpha_p_pred[80:220, :140, :]
    alpha_pc_truth = alpha_pc_truth[80:220, :140, :]
    alpha_pc_pred = alpha_pc_pred[80:220, :140, :]

    # Find a 9 element slice list containing the first and last ground truth predictions
    slice_list = get_slice_list(p_truth_volume, pc_truth_volume)

    fig, axs = plt.subplots(3, 3, figsize=(30, 30))

    for i, slice_idx in enumerate(slice_list):
        ax = axs[i // 3, i % 3]

        ax.imshow(mri_volume[:, :, slice_idx], cmap='gray')
        ax.axis('off')

        # Overlay ground truth and predicted patella masks
        ax.imshow(alpha_p_truth[:, :, slice_idx], cmap='Blues', alpha=alpha_p_truth[:, :, slice_idx])
        ax.imshow(alpha_p_pred[:, :, slice_idx], cmap='Reds', alpha=alpha_p_pred[:, :, slice_idx])

        # Overlay ground truth and predicted patellar cartilage masks
        ax.imshow(alpha_pc_truth[:, :, slice_idx], cmap='Wistia', alpha=alpha_pc_truth[:, :, slice_idx])
        ax.imshow(alpha_pc_pred[:, :, slice_idx], cmap='Greens', alpha=alpha_pc_pred[:, :, slice_idx])

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join("../results", model_name, f"{subj_name}_p_and_pc_windows.png")), dpi=600)
    plt.close()

    # MRI
    fig, axs = plt.subplots(9, 1, figsize=(45, 5))

    for i, slice_idx in enumerate(slice_list):
        ax = axs[i // 3, i % 3]
        ax = axs[i]

        ax.imshow(mri_volume[:, :, slice_idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join("../results", model_name, f"{subj_name}_mri_windows.png")), dpi=600)
    plt.close()

    # Patella only
    fig, axs = plt.subplots(9, 1, figsize=(45, 5))

    for i, slice_idx in enumerate(slice_list):
        ax = axs[i // 3, i % 3]
        ax = axs[i]

        ax.imshow(mri_volume[:, :, slice_idx], cmap='gray')
        ax.axis('off')

        # Overlay ground truth and predicted patella masks
        ax.imshow(alpha_p_truth[:, :, slice_idx], cmap='Blues', alpha=alpha_p_truth[:, :, slice_idx])
        ax.imshow(alpha_p_pred[:, :, slice_idx], cmap='Reds', alpha=alpha_p_pred[:, :, slice_idx])

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join("../results", model_name, f"{subj_name}_p_windows.png")), dpi=600)
    plt.close()

    # Patellar Cartilage Only
    fig, axs = plt.subplots(9, 1, figsize=(45, 5))

    for i, slice_idx in enumerate(slice_list):
        ax = axs[i // 3, i % 3]
        ax = axs[i]

        ax.imshow(mri_volume[:, :, slice_idx], cmap='gray')
        ax.axis('off')

        # Overlay ground truth and predicted patellar cartilage masks
        ax.imshow(alpha_pc_truth[:, :, slice_idx], cmap='Blues', alpha=alpha_pc_truth[:, :, slice_idx])
        ax.imshow(alpha_pc_pred[:, :, slice_idx], cmap='Reds', alpha=alpha_pc_pred[:, :, slice_idx])

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join("../results", model_name, f"{subj_name}_pc_windows.png")), dpi=600)
    plt.close()


def get_comparison_plot_filename(date_time):
    model_filename = get_model_filename(date_time)
    results_filename = build_results_filename(model_filename)
    comp_filename = os.path.join(results_filename, "comparisons")
    if not os.path.exists(comp_filename):
        os.mkdir(comp_filename)
    return comp_filename


def parse_dataset_name(model_name):
    """Returns dataset_name from the model name"""
    dataset_name = model_name.split("_")[-1]
    return dataset_name


def return_volumes(subj_name, model_name):
    """Returns volumes for a given subject and a given model"""
    # Get the dataset_name from model name
    dataset_name = parse_dataset_name(model_name)

    # Point to predictions
    cwd = os.getcwd()
    pred_folder = os.path.join(cwd, "results", model_name)
    p_pred_folder = os.path.join(pred_folder, "pat")
    pc_pred_folder = os.path.join(pred_folder, "pat_cart")

    # Point to Ground Truth
    truth_folder = get_data_path(dataset_name)
    mri_folder = os.path.join(truth_folder, "test", "mri")
    p_and_pc_truth_folder = os.path.join(truth_folder, "test", "mask")

    # Get list of images in each folder for this subject
    image_list = [f"{subj_name}-{four_digit_number(i)}.bmp" for i in range(26, 96)]

    # Append image list to all absolute paths to load
    p_pred_names = [os.path.join(p_pred_folder, image_name) for image_name in image_list]
    pc_pred_names = [os.path.join(pc_pred_folder, image_name) for image_name in image_list]
    mri_names = [os.path.join(mri_folder, image_name) for image_name in image_list]
    p_and_pc_truth_names = [os.path.join(p_and_pc_truth_folder, image_name) for image_name in image_list]

    # Load volumes
    p_pred_volume = assemble_mri_volume(p_pred_names)
    pc_pred_volume = assemble_mri_volume(pc_pred_names)
    mri_volume = assemble_mri_volume(mri_names)
    p_truth_volume, pc_truth_volume = assemble_mask_volume(p_and_pc_truth_names)

    return mri_volume, p_truth_volume, p_pred_volume, pc_truth_volume, pc_pred_volume


def get_most_recent_model():
    folder_path = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(folder_path, os.pardir))

    model_path = os.path.join(parent_dir, 'models')

    models = os.listdir(model_path)

    # Sort files by modification time
    models.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)

    date_times = list()

    if len(models) > 1:
        model = models[0].split(".")[0]
        date_times.append(model)  # Get most recent model without .h5 extension

        return date_times
    else:
        print("There are not enough files in the 'models' subfolder.")


def plot_loss(history, results_filename, show=False):
    plt.plot(history["val_loss"], label='val_loss')
    plt.plot(history["loss"], label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(results_filename, "loss.png"))
    plt.close()
    if show:
        plt.show()


def get_all_models_containing(substring):
    save_path = get_data_path(" ").split(os.path.sep)[0:-1]
    save_path = os.path.sep.join(save_path)
    model_path = os.path.join(save_path, "models")
    models = os.listdir(model_path)

    date_times = list()
    for model in models:
        if substring in model:
            date_times.append(model[:-3])
    return date_times


if __name__ == "__main__":

    # Mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # # date_time pattern to identify model we just trained
        date_times = get_most_recent_model()

        tissue = date_times[0][7]
        dataset_name = parse_dataset_name(date_times[0])

        date_times.append([f"unet3d-{tissue}_8888-88-88_88-88-88_{dataset_name}"])
        # date_times = ["unet_2024-07-11_00-40-25_ctHT5"]

        for date_time in date_times:

            # plot_mri_with_both_masks("AS_006", date_time)

            print(f"Evaluating model {date_time}")

            # Get results filename
            results_filename = get_results_filename(date_time)
            prep_results_filepath(results_filename)
            print(f"Saving results to {results_filename}")

            # get the history and model
            history, model = get_hist_and_model(date_time)

            # Create training curves from model and save them to the results filename
            plot_loss(history, results_filename, show=False)

            # Load in test dataset and create iterable
            test_dataset = get_dataset(dataset_name=dataset_name, dataset_type="test", batch_size=1, tissue=tissue)

            iterable = iter(test_dataset)
            n_test_scans = len(test_dataset)

            # Get comparison plots filename
            # comp_filename = get_comparison_plot_filename(date_time)

            # Count true pixels [intersection, predicted, true]
            pat_positives = [0, 0, 0]

            for i in range(n_test_scans):
                filename, mri, label = next(iterable)

                pred_label = model.predict(mri)

                mri = process_mri(mri)
                pat, pat_prob = process_predicted_label(pred_label)
                pat_true = process_true_label(label)

                pat_positives = count_positives(pat, pat_true, pat_positives)

                # Plot examples of true masks that have predictions on them
                # plot_mri_with_masks(mri, pat_true, pat, comp_filename, filename, tissue='pat')
                # plot_mri_with_masks(mri, pat_cart_true, pat_cart, comp_filename, filename, tissue='pat_cart')

                # Output predictions
                # save_result(filename, date_time, pat, pat_prob, tissue=tissue)

                # print(f"Img {i+1} of {n_test_images}")

            pat_dsc = calculate_dice(pat_positives)

            print(f"Model: {date_time}")
            print(f"Tissue: {parser.parse_args().tissue}")
            print(f"Patellar Dice Score: {pat_dsc}")

            metrics = {"dice": pat_dsc,
                       "pat_positive_counts": pat_positives,
                       "positive_count_info": ["intersection", "predicted", "true"]}

            save_model_info({"model_name": date_time,
                             "dice": pat_dsc})

            save_metrics(date_time, metrics)
