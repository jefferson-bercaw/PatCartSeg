import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from create_dataset import get_dataset
import datetime
from dice_loss_function import dice_loss


def get_date_and_hour():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H")
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


def process_label(pred_label):
    thresholded_label = pred_label >= 0.5
    binary_data = thresholded_label.astype(np.uint8)

    pat = np.squeeze(binary_data[:, :, :, 0])
    pat_cart = np.squeeze(binary_data[:, :, :, 1])

    return pat, pat_cart


def process_mri(mri):
    mri = np.squeeze(mri)
    return mri


if __name__ == "__main__":
    # date_time pattern to identify model we just trained
    num_examples = 100
    date_time = get_date_and_hour()

    # Get results filename
    results_filename = get_results_filename(date_time)
    prep_results_filepath(results_filename)

    # get the history and model
    history, model = get_hist_and_model(date_time)
    test_dataset = get_dataset(batch_size=1, dataset_type='test')

    # Get evaluation metrics
    # loss, accuracy = model.evaluate(test_dataset)

    # Get predicted label metrics
    # predicted_labels = model.predict(test_dataset)
    # predicted_labels = tf.argmax(predicted_labels, axis=1)

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
    plt.show()
    plt.savefig(results_filename + "\\tn.png")

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
    for i in range(num_examples):
        mri, label = next(iterable)
        pred_label = model.predict(mri)

        mri = process_mri(mri)
        pat, pat_cart = process_label(pred_label)

        fig, axs = plt.subplots(3, 1)

        axs[0].imshow(mri, cmap='gray')
        axs[0].set_title("MRI")

        axs[1].imshow(pat, cmap='gray')
        axs[1].set_title("Predicted Patella")

        axs[2].imshow(pat_cart, cmap='gray')
        axs[2].set_title("Predicted Patellar Cartilage")

        plt.tight_layout()

        plt.savefig(results_filename + "\\examples\\" + str(i) + ".png", dpi=600)
        plt.show()



