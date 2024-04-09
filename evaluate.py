import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from create_dataset import get_dataset
import datetime


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
    model = keras.models.load_model(model_filename)
    return model


def get_hist_and_model(date_time):
    history_filename = get_history_filename(date_time)
    model_filename = get_model_filename(date_time)
    history = load_history(history_filename)
    model = load_model(model_filename)
    return history, model


if __name__ == "__main__":
    # date_time pattern to identify model we just trained
    date_time = get_date_and_hour()

    # get the history and model
    history, model = get_hist_and_model(date_time)
    test_dataset = get_dataset(batch_size=1, dataset_type='test')

    # Get evaluation metrics
    loss, accuracy = model.evaluate(test_dataset)

    # Get predicted label metrics
    predicted_labels = model.predict(test_dataset)
    predicted_labels = tf.argmax(predicted_labels, axis=1)

    # Output plots
    plt.plot(history["FN"])
    plt.xlabel('Epoch')
    plt.title("False Negatives")
    plt.show()

    plt.plot(history["FP"])
    plt.xlabel('Epoch')
    plt.title("False Positives")
    plt.show()

    plt.plot(history["TN"])
    plt.xlabel('Epoch')
    plt.title("True Negatives")
    plt.show()

    plt.plot(history["TP"])
    plt.xlabel('Epoch')
    plt.title("True Positives")
    plt.show()

    plt.plot(history["val_loss"], label='val_loss')
    plt.plot(history["loss"], label='train_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title("Loss")
    plt.show()

