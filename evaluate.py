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


def get_date_and_hour():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    return date_time


def get_history_filename(date_time):
    files = os.listdir("./history")
    for filename in files:
        if date_time in filename:
            history_filename = os.path.abspath(os.path.join('./history', filename))
            return history_filename


def get_model_filename(date_time):
    files = os.listdir("./models")
    for filename in files:
        if date_time in filename:
            model_filename = os.path.abspath(os.path.join('./models', filename))
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
    files = results_directory.split("\\")
    files[-2] = 'results'

    separator = "\\"
    results_filename = separator.join(files)
    return results_filename


def get_hist_and_model(date_time):
    history_filename = get_history_filename(date_time)
    model_filename = get_model_filename(date_time)
    history = load_history(history_filename)
    model = load_model(model_filename)
    return history, model


def get_results_filename(date_time):
    model_filename = get_model_filename(date_time)
    results_filename = build_results_filename(model_filename)
    return results_filename


def prep_results_filepath(results_filename):
    if not os.path.exists(results_filename):
        os.mkdir(results_filename)

    examples_filename = results_filename + "\\examples"
    if not os.path.exists(examples_filename):
        os.mkdir(examples_filename)


def process_predicted_label(pred_label):
    thresholded_label = (pred_label >= 0.5)
    binary_data = thresholded_label.astype(np.uint8)

    pat = np.squeeze(binary_data[:, :, :, 0])
    pat_cart = np.squeeze(binary_data[:, :, :, 1])

    return pat, pat_cart


def process_mri(mri):
    mri = np.squeeze(mri)
    return mri


def save_result(filename, date_time, pat, pat_cart):
    filename_str = filename.numpy()[0].decode()
    results_filename = get_results_filename(date_time)

    pat_filepath = results_filename + "\\pat"
    pat_cart_filepath = results_filename + "\\pat_cart"

    # Make directories if they don't exist
    if not os.path.exists(pat_filepath):
        os.mkdir(pat_filepath)
    if not os.path.exists(pat_cart_filepath):
        os.mkdir(pat_cart_filepath)

    pat_filepath = pat_filepath + "\\" + filename_str
    pat_cart_filepath = pat_cart_filepath + "\\" + filename_str

    pat_img = Image.fromarray(pat)
    pat_cart_img = Image.fromarray(pat_cart)

    # Save the image as a BMP file
    pat_img.save(pat_filepath)
    pat_cart_img.save(pat_cart_filepath)
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
    with open(results_filename + "\\metrics.pkl", 'wb') as f:
        pickle.dump(metrics, f)
    return


def plot_mri_with_masks(mri_image, ground_truth_mask, predicted_mask):
    # Define colors for ground truth and predicted masks
    gt_color = 'blue'
    pred_color = 'red'
    overlap_color = 'purple'  # Color for areas of overlap

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot MRI image
    ax.imshow(mri_image, cmap='gray')

    # Overlay ground truth mask
    ax.imshow(ground_truth_mask*255, cmap='Blues', alpha=0.5)

    # Overlay predicted mask
    ax.imshow(predicted_mask*255, cmap='Reds', alpha=0.5)

    # Create legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=gt_color, alpha=0.5, label='Ground Truth'),
        plt.Rectangle((0, 0), 1, 1, color=pred_color, alpha=0.5, label='Predicted')
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    plt.show()


if __name__ == "__main__":
    # date_time pattern to identify model we just trained
    num_examples = 100
    # date_time = get_date_and_hour()
    date_time = "2024-04-17_08"

    # Get results filename
    results_filename = get_results_filename(date_time)
    prep_results_filepath(results_filename)

    # get the history and model
    history, model = get_hist_and_model(date_time)
    test_dataset = get_dataset(batch_size=1, dataset_type='test')

    # Output plots
    plt.plot(history["FN"])
    plt.xlabel('Epoch')
    plt.title("False Negatives")
    plt.savefig(results_filename + "\\fn.png")
    plt.show()

    plt.plot(history["FP"])
    plt.xlabel('Epoch')
    plt.title("False Positives")
    plt.savefig(results_filename + "\\fp.png")
    plt.show()

    plt.plot(history["TN"])
    plt.xlabel('Epoch')
    plt.title("True Negatives")
    plt.savefig(results_filename + "\\tn.png")
    plt.show()

    plt.plot(history["TP"])
    plt.xlabel('Epoch')
    plt.title("True Positives")
    plt.savefig(results_filename + "\\tp.png")
    plt.show()

    plt.plot(history["val_loss"], label='val_loss')
    plt.plot(history["loss"], label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig(results_filename + "\\loss.png")
    plt.show()

    iterable = iter(test_dataset)
    n_test_images = len(test_dataset)

    # Count true pixels [intersection, predicted, true]
    pat_positives = [0, 0, 0]
    pat_cart_positives = [0, 0, 0]
    example_num = 0

    for i in range(n_test_images):
        filename, mri, label = next(iterable)

        pred_label = model.predict(mri)

        mri = process_mri(mri)
        pat, pat_cart = process_predicted_label(pred_label)
        pat_true, pat_cart_true = process_true_label(label)

        pat_positives = count_positives(pat, pat_true, pat_positives)
        pat_cart_positives = count_positives(pat_cart, pat_cart_true, pat_cart_positives)

        # Plot examples of true masks that have predictions on them
        if np.sum(pat_true) > 0 and np.sum(pat_cart_true) > 0 and example_num < num_examples:
            plot_mri_with_masks(mri, pat_true, pat)
            plot_mri_with_masks(mri, pat_cart_true, pat_cart)
            example_num += 1

        # save_result(filename, date_time, pat, pat_cart)
        print(f"Img {i} of {n_test_images}")

    pat_dsc = calculate_dice(pat_positives)
    pat_cart_dsc = calculate_dice(pat_cart_positives)

    print(f"Patellar Dice Score: {pat_dsc}")
    print(f"Patellar Cartilage Dice Score: {pat_cart_dsc}")

    metrics = {"patellar_dice": pat_dsc,
               "patellar_cartilage_dice": pat_cart_dsc,
               "pat_positive_counts": pat_positives,
               "pat_cart_positive_counts": pat_cart_positives,
               "positive_count_info": ["intersection", "predicted", "true"]}
    save_metrics(date_time, metrics)

