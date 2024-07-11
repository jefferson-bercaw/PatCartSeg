import os
import shutil
from get_data_path import get_data_path


def get_slice_num(file):
    """Returns integer of 1 - 120 of slice number"""
    end_str = file.split("-")[-1]
    num_str = end_str.split(".")[0]
    return int(num_str)


def move_all(data_path, save_data_path, to_exclude):
    """Moving all contents from the data_path to the save_data_path within to_exclude bounds"""
    files = os.listdir(data_path)
    for file in files:
        slice_num = get_slice_num(file)
        if (slice_num > to_exclude // 2) and (slice_num <= 120 - to_exclude // 2):
            source_path = os.path.join(data_path, file)
            dest_path = os.path.join(save_data_path, file)
            shutil.copyfile(source_path, dest_path)
    return


if __name__ == "__main__":
    data_path = get_data_path("Paranjape_Cropped")
    save_data_path = get_data_path("Paranjape_ct")
    move_all(data_path, save_data_path, to_exclude=50)
