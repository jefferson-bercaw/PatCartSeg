import numpy as np
import pickle
import os
from unet import build_unet
from dice_loss_function import dice_loss
from tensorflow import keras
import matplotlib.pyplot as plt
from evaluate import plot_mri_with_both_masks


if __name__ == "__main__":
    # Histogram Figure
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
    plt.savefig("R:/DefratePrivate/Bercaw/Patella_Autoseg/results/motion_histogram.png", dpi=600)
    plt.show()

    # Output Metrics
    model_list = ["unet_2024-04-26_19-09-53_task0", "unet_2024-04-27_02-08-42_task1", "unet_2024-04-27_10-04-35_task2",
                  "unet_2024-04-27_00-16-29_task3", "unet_2024-04-27_04-36-06_task4", "unet_2024-04-26_15-07-55_task5"]
    motion_labels = ['Minimal', 'Mild', 'Moderate', 'Significant', 'Severe', 'No Motion']
    table = list()
    for idx, (model, label) in enumerate(zip(model_list, motion_labels)):
        # Opening metrics file
        with open(f'./results/{model}/metrics.pkl', 'rb') as file:
            metrics = pickle.load(file)
            file.close()

        with open(f'./history/{model}.pkl', 'rb') as file:
            hist = pickle.load(file)
            file.close()

        pd = metrics['patellar_dice']
        pcd = metrics['patellar_cartilage_dice']
        epochs = len(hist['FP'])

        table.append([label, epochs, pd, pcd])

    print(table)

