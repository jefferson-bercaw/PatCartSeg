import os
from PIL import Image
from get_data_path import get_data_path
import numpy as np
import pickle

if __name__ == "__main__":

    data_path = get_data_path("HT")
    dataset_types = ["test", "train", "val"]

    max_rows = [0, 0]  # P, PC
    max_cols = [0, 0]

    min_rows = [500, 500]
    min_cols = [500, 500]

    for dataset_type in dataset_types:
        file_path = os.path.join(data_path, dataset_type, "mask")
        files = os.listdir(file_path)

        for idx, file in enumerate(files):
            file_name = os.path.join(data_path, dataset_type, "mask", file)

            img = Image.open(file_name)
            img_array = np.array(img)

            P_row, P_col = np.nonzero(img_array == 1)
            PC_row, PC_col = np.nonzero(img_array == 2)

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

            # Print
            if idx % 100 == 0:
                print(f"File {idx} of {len(files)} in {dataset_type}")

    P_box = [min_cols[0], max_cols[0], min_rows[0], max_rows[0]]
    PC_box = [min_cols[1], max_cols[1], min_rows[1], max_rows[1]]
    box_info = ["Min_Column (left)", "Max_Column (right)", "Min_Row (top)", "Max_Row (bottom"]

    P_size = [max_cols[0] - min_cols[0] + 1, max_rows[0] - min_rows[0] + 1]
    PC_size = [max_cols[1] - min_cols[1] + 1, max_rows[1] - min_rows[1] + 1]
    size_info = ["X_pixels", "Y_pixels"]

    with open("results/size_info_HT.pkl", "wb") as f:
        pickle.dump((P_box, PC_box, box_info, P_size, PC_size, size_info), f)


