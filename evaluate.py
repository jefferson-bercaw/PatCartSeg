import matplotlib.colors
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from create_dataset import get_dataset
import datetime
from dice_loss_function import dice_loss
from PIL import Image
from unet import build_unet
from get_data_path import get_data_path
from augmentation import assemble_mask_volume, assemble_mri_volume, four_digit_number


def get_date_and_hour():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    return date_time


def get_history_filename(date_time):
    files = os.listdir("history")
    for filename in files:
        if date_time[0:23] in filename:
            history_filename = os.path.abspath(os.path.join('history', filename))
            return history_filename


def get_model_filename(date_time):
    # files = os.listdir("models")
    # for filename in files:
    #     if date_time is filename:
    model_filename = os.path.abspath(os.path.join('models', date_time))
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

    if "\\" in results_directory:
        files = results_directory.split("\\")
        separator = "\\"
    elif "/" in results_directory:
        files = results_directory.split("/")
        separator = "/"
    else:
        raise(SyntaxError, f"There are no single forward or double backward slashes in the results directory: {results_directory}")

    files[-2] = 'results'
    results_filename = separator.join(files)
    return results_filename


def get_hist_and_model(date_time):
    if "lowest_val" not in date_time:
        history_filename = get_history_filename(date_time)
        history = load_history(history_filename)
    else:
        history = None

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

    pat = np.squeeze(binary_data[:, :, :, 0])
    pat_cart = np.squeeze(binary_data[:, :, :, 1])

    # Probability masks
    pat_prob = np.squeeze(pred_label[:, :, :, 0])
    pat_cart_prob = np.squeeze(pred_label[:, :, :, 1])

    return pat, pat_cart, pat_prob, pat_cart_prob


def process_mri(mri):
    mri = np.squeeze(mri)
    return mri


def save_result(filename, date_time, pat, pat_cart, pat_prob, pat_cart_prob):
    filename_str = filename.numpy()[0].decode()
    results_filename = get_results_filename(date_time)

    pat_filepath = os.path.join(results_filename, "pat")
    pat_cart_filepath = os.path.join(results_filename, "pat_cart")

    pat_prob_filepath = os.path.join(results_filename, "pat_prob")
    pat_cart_prob_filepath = os.path.join(results_filename, "pat_cart_prob")

    # Make directories if they don't exist
    if not os.path.exists(pat_filepath):
        os.mkdir(pat_filepath)
    if not os.path.exists(pat_cart_filepath):
        os.mkdir(pat_cart_filepath)
    if not os.path.exists(pat_prob_filepath):
        os.mkdir(pat_prob_filepath)
    if not os.path.exists(pat_cart_prob_filepath):
        os.mkdir(pat_cart_prob_filepath)

    filename_npy = filename_str.split('.')[0] + ".npy"

    pat_filepath = os.path.join(pat_filepath, filename_str)
    pat_cart_filepath = os.path.join(pat_cart_filepath, filename_str)
    pat_prob_filepath = os.path.join(pat_prob_filepath, filename_npy)
    pat_cart_prob_filepath = os.path.join(pat_cart_prob_filepath, filename_npy)
    print(f"Saving numpy arrays to {pat_prob_filepath} and {pat_cart_prob_filepath}")

    # Save the masks as BMP files
    pat_img = Image.fromarray(pat)
    pat_cart_img = Image.fromarray(pat_cart)
    pat_img.save(pat_filepath)
    pat_cart_img.save(pat_cart_filepath)

    # Save the probability masks as NPY files
    np.save(pat_prob_filepath, pat_prob)
    np.save(pat_cart_prob_filepath, pat_cart_prob)

    return


def process_true_label(label):
    label = tf.squeeze(label, axis=0)
    pat = label[:, :, 0].numpy().astype(np.uint8)
    pat_cart = label[:, :, 1].numpy().astype(np.uint8)
    return pat, pat_cart


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
    for i in range(120):
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

    alpha_p_truth = np.where(p_truth_volume == 1, 0.4, 0)  # Set alpha based on mask values
    alpha_p_pred = np.where(p_pred_volume == 1, 0.4, 0)  # Set alpha based on mask values
    alpha_pc_truth = np.where(pc_truth_volume == 1, 0.4, 0)
    alpha_pc_pred = np.where(pc_pred_volume == 1, 0.4, 0)

    top_left_coords = [150, 50]  # row, col
    img_size = 175
    bottom_right_coords = [top_left_coords[0] + img_size, top_left_coords[1] + img_size]

    # Find a 9 element slice list containing the first and last ground truth predictions
    slice_list = get_slice_list(p_truth_volume, pc_truth_volume)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, slice_idx in enumerate(slice_list):
        ax = axs[i // 3, i % 3]

        ax.imshow(mri_volume[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='gray')
        ax.axis('off')

        # Overlay ground truth and predicted patella masks
        ax.imshow(alpha_p_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Blues', alpha=alpha_p_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])
        ax.imshow(alpha_p_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Reds', alpha=alpha_p_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])

        # Overlay ground truth and predicted patellar cartilage masks
        ax.imshow(alpha_pc_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Wistia', alpha=alpha_pc_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])
        ax.imshow(alpha_pc_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Greens', alpha=alpha_pc_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join("results", model_name, f"{subj_name}_p_and_pc_windows.png")), dpi=600)
    plt.show()

    # Patella only
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, slice_idx in enumerate(slice_list):
        ax = axs[i // 3, i % 3]

        ax.imshow(mri_volume[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='gray')
        ax.axis('off')

        # Overlay ground truth and predicted patella masks
        ax.imshow(alpha_p_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Blues', alpha=alpha_p_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])
        ax.imshow(alpha_p_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Reds', alpha=alpha_p_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join("results", model_name, f"{subj_name}_p_windows.png")), dpi=600)
    plt.show()

    # Patellar Cartilage Only
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, slice_idx in enumerate(slice_list):
        ax = axs[i // 3, i % 3]

        ax.imshow(mri_volume[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='gray')
        ax.axis('off')

        # Overlay ground truth and predicted patellar cartilage masks
        ax.imshow(alpha_pc_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Blues', alpha=alpha_pc_truth[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])
        ax.imshow(alpha_pc_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx], cmap='Reds', alpha=alpha_pc_pred[top_left_coords[0]:bottom_right_coords[0], top_left_coords[1]:bottom_right_coords[1], slice_idx])

    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join("results", model_name, f"{subj_name}_pc_windows.png")), dpi=600)
    plt.show()


def get_comparison_plot_filename(date_time):
    model_filename = get_model_filename(date_time)
    results_filename = build_results_filename(model_filename)
    comp_filename = os.path.join(results_filename, "comparisons")
    if not os.path.exists(comp_filename):
        os.mkdir(comp_filename)
    return comp_filename


def parse_dataset_name(model_name):
    """Returns dataset_name from the model name of rotation and translations"""
    no_ext = model_name[:-3]  # Removes .h5
    underscore_strs = no_ext.split("_")
    dataset_name = underscore_strs[-3] + "_" + underscore_strs[-2] + "_" + underscore_strs[-1]
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
    image_list = [f"{subj_name}-{four_digit_number(i)}.bmp" for i in range(1, 120)]

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


def get_most_recent_models():
    folder_path = os.getcwd()
    model_path = os.path.join(folder_path, 'models')

    models = os.listdir(model_path)

    # Sort files by modification time
    models.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)

    date_times = list()

    if len(models) > 1:
        date_times.append(models[0])  # Lowest validation loss
        date_times.append(models[1])  # End (after early stopping condition)
        return date_times
    else:
        print("There are not enough files in the 'models' subfolder.")


if __name__ == "__main__":

    # Mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # # date_time pattern to identify models we just trained
        # date_times = get_most_recent_models()
        #
        # # not analyzing lowest_val_loss model
        # date_times = [date_time for date_time in date_times if "lowest" not in date_time]
        date_times = ["unet_2024-05-25_10-23-16_cHT5_30_5.h5",
                      "unet_2024-05-25_03-50-51_cHT5_30_10.h5"]

        # print(f"Most recent models being analyzed: {date_times}")
        # date_time = get_date_and_hour()
        # plot_mri_with_both_masks(subj_name, model_name)

        for date_time in date_times:

            print(f"Evaluating model {date_time}")

            dataset_name = parse_dataset_name(date_time)

            # Get results filename
            results_filename = get_results_filename(date_time)
            prep_results_filepath(results_filename)
            print(f"Saving results to {results_filename}")

            # get the history and model
            history, model = get_hist_and_model(date_time)

            test_dataset = get_dataset(batch_size=1, dataset_type='test', dataset=dataset_name)

            if history is not None:
                # Output plots
                plt.plot(history["FN"])
                plt.xlabel('Epoch')
                plt.title("False Negatives")
                plt.savefig(os.path.join(results_filename, "fn.png"))
                # plt.show()

                plt.plot(history["FP"])
                plt.xlabel('Epoch')
                plt.title("False Positives")
                plt.savefig(os.path.join(results_filename, "fp.png"))
                # plt.show()

                plt.plot(history["TN"])
                plt.xlabel('Epoch')
                plt.title("True Negatives")
                plt.savefig(os.path.join(results_filename, "tn.png"))
                # plt.show()

                plt.plot(history["TP"])
                plt.xlabel('Epoch')
                plt.title("True Positives")
                plt.savefig(os.path.join(results_filename, "tp.png"))
                # plt.show()

                plt.plot(history["val_loss"], label='val_loss')
                plt.plot(history["loss"], label='train_loss')
                plt.xlabel('Epoch')
                plt.ylabel('Dice Loss')
                plt.legend()
                plt.title("Loss")
                plt.savefig(os.path.join(results_filename, "loss.png"))
                # plt.show()

            iterable = iter(test_dataset)
            n_test_images = len(test_dataset)

            # Get comparison plots filename
            comp_filename = get_comparison_plot_filename(date_time)

            # Count true pixels [intersection, predicted, true]
            pat_positives = [0, 0, 0]
            pat_cart_positives = [0, 0, 0]

            for i in range(n_test_images):
                filename, mri, label = next(iterable)

                pred_label = model.predict(mri)

                mri = process_mri(mri)
                pat, pat_cart, pat_prob, pat_cart_prob = process_predicted_label(pred_label)
                pat_true, pat_cart_true = process_true_label(label)

                pat_positives = count_positives(pat, pat_true, pat_positives)
                pat_cart_positives = count_positives(pat_cart, pat_cart_true, pat_cart_positives)

                # Plot examples of true masks that have predictions on them
                plot_mri_with_masks(mri, pat_true, pat, comp_filename, filename, tissue='pat')
                plot_mri_with_masks(mri, pat_cart_true, pat_cart, comp_filename, filename, tissue='pat_cart')

                # Output predictions
                save_result(filename, date_time, pat, pat_cart, pat_prob, pat_cart_prob)

                print(f"Img {i} of {n_test_images}")

            pat_dsc = calculate_dice(pat_positives)
            pat_cart_dsc = calculate_dice(pat_cart_positives)

            print(f"Model: {date_time}")
            print(f"Patellar Dice Score: {pat_dsc}")
            print(f"Patellar Cartilage Dice Score: {pat_cart_dsc}")

            metrics = {"patellar_dice": pat_dsc,
                       "patellar_cartilage_dice": pat_cart_dsc,
                       "pat_positive_counts": pat_positives,
                       "pat_cart_positive_counts": pat_cart_positives,
                       "positive_count_info": ["intersection", "predicted", "true"]}

            save_metrics(date_time, metrics)
