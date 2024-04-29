import numpy as np
import pickle
import os
from unet import build_unet
from dice_loss_function import dice_loss
from tensorflow import keras
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open('./augmentation/new_parameters.pkl', 'rb') as file:
        aug_dict = pickle.load(file)
        file.close()

    motion_labels = [["Minimal", 'green'], ["Mild", 'blue'], ["Moderate", 'Gold'], ["Significant", 'darkorange'], ["Severe", 'red']]
    for mean_val, sigma, motion_label in zip(aug_dict['mean_vals'], aug_dict['sigmas'], motion_labels):

        sigma_array = np.array(sigma)

        # Get movement in each axis
        movement_X = sigma_array[:, 0]
        movement_Y = sigma_array[:, 1]
        movement_Z = sigma_array[:, 2]

        plt.hist(movement_X, bins=12, alpha=0.5, label=motion_label[0], color=motion_label[1])

    plt.xlabel("Blurring Factor $\sigma$")
    plt.legend(loc='best')
    plt.title("Motion artifact distributions")
    plt.show()

    print("Done")
