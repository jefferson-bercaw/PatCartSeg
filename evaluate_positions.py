import os
from PIL import Image
from get_data_path import get_data_path
import numpy as np
import pickle
import matplotlib.pyplot as plt

def perform_positional_analyses():

    data_path = get_data_path("HTCO")
    dataset_types = ["test", "train", "val"]
    bounds_dict = {}

    for dataset_type in dataset_types:
        file_path = os.path.join(data_path, dataset_type, "mask")
        files = os.listdir(file_path)

        # Sort files alphabetically
        files.sort()

        for idx, file in enumerate(files):
            file_name = os.path.join(data_path, dataset_type, "mask", file)

            img = Image.open(file_name)
            img_array = np.array(img)

            # Row and Column pairs of row and column numbers containing true elements for that tissue
            P_row, P_col = np.nonzero(img_array == 1)
            PC_row, PC_col = np.nonzero(img_array == 2)

            # Get slice number
            slice_num = int(file.split("-")[1].split(".")[0])

            # Reset max and min values for this subject
            if slice_num == 1:
                max_rows = [0, 0]
                max_cols = [0, 0]

                min_rows = [500, 500]
                min_cols = [500, 500]

                slices = [0, 121]

            # Maximum
            if P_row.size > 0 and np.max(P_row) > max_rows[0]:
                max_rows[0] = np.max(P_row)

            if P_col.size > 0 and np.max(P_col) > max_cols[0]:
                max_cols[0] = np.max(P_col)

            if PC_row.size > 0 and np.max(PC_row) > max_rows[1]:
                max_rows[1] = np.max(PC_row)

            if PC_col.size > 0 and np.max(PC_col) > max_cols[1]:
                max_cols[1] = np.max(PC_col)

            # Minimum
            if P_row.size > 0 and np.min(P_row) < min_rows[0]:
                min_rows[0] = np.min(P_row)

            if P_col.size > 0 and np.min(P_col) < min_cols[0]:
                min_cols[0] = np.min(P_col)

            if PC_row.size > 0 and np.min(PC_row) < min_rows[1]:
                min_rows[1] = np.min(PC_row)

            if PC_col.size > 0 and np.min(PC_col) < min_cols[1]:
                min_cols[1] = np.min(PC_col)

            # Set max and min slices for this subject
            if (P_row.size > 0 or PC_row.size > 0) and slices[0] == 0:
                slices[0] = slice_num
            if (P_row.size > 0 or PC_row.size > 0):
                slices[1] = slice_num

            # Save subject info to a dict
            if slice_num == 120:
                top_right = [np.min([min_rows[0], min_rows[1]]), np.max([max_cols[0], max_cols[1]])]  # row, col
                bottom_left = [np.max([max_rows[0], max_rows[1]]), np.min([min_cols[0], min_cols[1]])]  # row, col

                slice_bounds = [slices[0], slices[1]]

                subj_id = file.split("-")[0]
                bounds_dict[subj_id] = {}

                bounds_dict[subj_id]["top_right"] = top_right
                bounds_dict[subj_id]["bottom_left"] = bottom_left
                bounds_dict[subj_id]["slice_bounds"] = slice_bounds

                bounds_dict[subj_id]["row_size"] = -top_right[0] + bottom_left[0] - 1
                bounds_dict[subj_id]["col_size"] = top_right[1] - bottom_left[1] + 1
                bounds_dict[subj_id]["slice_size"] = slice_bounds[1] - slice_bounds[0] + 1

                print("Subject:", subj_id)
    # P_box = [min_cols[0], max_cols[0], min_rows[0], max_rows[0]]
    # PC_box = [min_cols[1], max_cols[1], min_rows[1], max_rows[1]]
    # box_info = ["Min_Column (left)", "Max_Column (right)", "Min_Row (top)", "Max_Row (bottom"]
    #
    # P_size = [max_cols[0] - min_cols[0] + 1, max_rows[0] - min_rows[0] + 1]
    # PC_size = [max_cols[1] - min_cols[1] + 1, max_rows[1] - min_rows[1] + 1]
    # size_info = ["X_pixels", "Y_pixels"]

    # with open("results/size_info_HT.pkl", "wb") as f:
    #     pickle.dump((P_box, PC_box, box_info, P_size, PC_size, size_info), f)

    with open("results/bounds_dict_HTCO.pkl", "wb") as f:
        pickle.dump(bounds_dict, f)


def analyze_positions():
    # Read in dict
    with open("results/bounds_dict_HTCO.pkl", "rb") as f:
        bounds_dict = pickle.load(f)

    # get row, column, and slice max sizes
    row_sizes = []
    col_sizes = []
    slice_sizes = []
    for key in bounds_dict.keys():
        row_sizes.append(bounds_dict[key]["row_size"])
        col_sizes.append(bounds_dict[key]["col_size"])
        slice_sizes.append(bounds_dict[key]["slice_size"])
    max_row_size = max(row_sizes)
    max_col_size = max(col_sizes)
    max_slice_size = max(slice_sizes)

    # Print
    print("Max Row Size:", max_row_size)
    print("Max Column Size:", max_col_size)
    print("Max Slice Size:", max_slice_size)

    # Plot histogram of sizes
    plt.hist(row_sizes, bins=20)
    plt.xlabel("Row Size")
    plt.ylabel("Frequency")
    plt.title("Row Size Distribution")
    plt.show()

    plt.hist(col_sizes, bins=20)
    plt.xlabel("Column Size")
    plt.ylabel("Frequency")
    plt.title("Column Size Distribution")
    plt.show()

    plt.hist(slice_sizes, bins=20)
    plt.xlabel("Slice Size")
    plt.ylabel("Frequency")
    plt.title("Slice Size Distribution")
    plt.show()

    print()

if __name__ == "__main__":
    perform_positional_analyses()
    analyze_positions()
