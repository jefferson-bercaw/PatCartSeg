import pickle
import numpy as np
from get_data_path import get_data_path
import os
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("results/size_info_HT.pkl", "rb") as f:
        data = pickle.load(f)

    P_box = data[0]
    PC_box = data[1]

    # Min and max rows and columns of the patella and patellar cartilage
    x = [min([P_box[0], PC_box[0]]), max([P_box[1], PC_box[1]])]
    y = [min([P_box[2], PC_box[2]]), max([P_box[3], PC_box[3]])]

    # xy_dim of cropped image
    xy_dim = 256

    # Size of x and y span of P and PC
    x_sz = x[1] - x[0] + 1
    y_sz = y[1] - y[0] + 1

    # Buffers on either side of the bounding box
    x_buffer_size = (xy_dim - x_sz) // 2
    y_buffer_size = (xy_dim - y_sz) // 2

    # x and y starting coords (top left) and ending coords (bottom right)
    x_start = x[0] - x_buffer_size
    y_start = y[0] - y_buffer_size

    # Correct for negative values
    if x_start < 0:
        x_start = 0
    if y_start < 0:
        y_start = 0

    x_end = x_start + xy_dim
    y_end = y_start + xy_dim

    # iterate through each image and crop
    data_path = get_data_path("HT")
    dataset_types = ["test", "train", "val"]

    for dataset_type in dataset_types:
        file_path = os.path.join(data_path, dataset_type, "mask")
        files = os.listdir(file_path)

        for idx, file in enumerate(files):
            mask_name = os.path.join(data_path, dataset_type, "mask", file)
            mri_name = os.path.join(data_path, dataset_type, "mri", file)

            mask_img = Image.open(mask_name)
            mask = np.array(mask_img)

            mri_img = Image.open(mri_name)
            mri = np.array(mri_img)

            # Crop
            mask_crop = mask[y_start:y_end, x_start:x_end]
            mri_crop = mri[y_start:y_end, x_start:x_end]

            plt.imshow(mask_crop, cmap='gray')
            plt.show()

            print()