import os
import numpy as np
from PIL import Image
import random
from get_data_path import get_data_path


def get_all_scan_names(directory):
    folders = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)) and not folder.startswith("."):
            folders.append(folder)
    return folders


def get_excluded_names(filepath):
    exclude_list = []
    with open(filepath, 'r') as file:
        for line in file:
            folder_name = line.strip()
            exclude_list.append(folder_name)
    return exclude_list


def remove_excluded_scans(scans_list, exclude_list):
    if not exclude_list:  # if there are no scans to exclude
        return scans_list
    else:
        scans_list = [folder for folder in scans_list if folder not in exclude_list]
        return scans_list


def organize_subject_scans(scans_list):
    d = {}
    for scan in scans_list:
        AS, num = scan.split("_")
        num = int(num)

        if num <= 47:
            subj_num = int(np.floor(num / 6))

        elif num <= 77:
            subj_num = int(np.floor((num-48) / 2)) + 9

        elif num <= 97:
            subj_num = int(np.floor((num-78) / 2)) + 24

        elif num<= 129:
            subj_num = int(np.floor((num-98) / 4)) + 34

        d[scan] = {
            "scan": scan,
            "scan_num": num,
            "subject_num": subj_num
        }

    return d


def read_bmp(image_path, image_name):
    bmp_image = Image.open(os.path.join(image_path, image_name))
    np_image = np.array(bmp_image)
    return np_image


def assemble_2d_mask(P_mask, PC_mask):
    mask = np.zeros((512, 512))
    mask[P_mask > 0] = 1
    mask[PC_mask > 0] = 2

    return mask


def get_bmp_files(folder_path):
    bmp_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp"):
            bmp_files.append(filename)
    return bmp_files


def save_bmp(np_array, file_location):
    im = Image.fromarray(np_array)
    im.save(file_location)


if __name__ == "__main__":
    # Save option
    save_opt = True

    # Data directory
    data_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Organized_Data"

    # Get lists of each scan
    scans_list = get_all_scan_names(data_dir)
    exclude_list = get_excluded_names("data_exclude.txt")
    scans_list = remove_excluded_scans(scans_list, exclude_list)

    # Get Subject IDs from each scan
    scans = organize_subject_scans(scans_list)
    max_subject_num = max(entry['subject_num'] for entry in scans.values())

    # Randomly select subjects for test and subjects for validation
    random.seed(42)

    num_val_subj = 6
    num_test_subj = 6

    val_subj_nums = random.sample(range(1, max_subject_num+1), num_val_subj)
    test_subj_nums = random.sample([num for num in range(1, max_subject_num+1) if num not in val_subj_nums], num_test_subj)

    # Save these images in one combined folder
    train_scans = [entry["scan"] for entry in scans.values() if (entry['subject_num'] not in test_subj_nums and entry['subject_num'] not in val_subj_nums)]
    test_scans = [entry["scan"] for entry in scans.values() if entry['subject_num'] in test_subj_nums]
    val_scans = [entry["scan"] for entry in scans.values() if entry['subject_num'] in val_subj_nums]

    # Get destinations of where we're saving each group
    dataset_path = get_data_path(dataset="HTCO")
    dest_train = dataset_path + "/train"
    dest_test = dataset_path + "/test"
    dest_val = dataset_path + "/val"

    for scan in scans.keys():

        source_path = os.path.join(data_dir, scan)

        source_mri = os.path.join(source_path, "mri")
        source_P = os.path.join(source_path, "P")
        source_PC = os.path.join(source_path, "PC")

        # List of all image files
        image_files = get_bmp_files(source_mri)

        # Open files, assemble 3d masks, save mri and mask as npz files
        print(f"Saving scan {scan}")

        for file in image_files:

            mri = read_bmp(source_mri, file)
            P = read_bmp(source_P, file)
            PC = read_bmp(source_PC, file)

            mask = assemble_2d_mask(P, PC)
            mask = mask.astype(np.uint8)
            file_to_save = file.split(".")[0] + ".bmp"

            # Save .bmp files
            if save_opt:
                if scan in train_scans:
                    save_bmp(mri, str(dest_train + "/mri/" + file_to_save))
                    save_bmp(mask, str(dest_train + "/mask/" + file_to_save))

                elif scan in test_scans:
                    save_bmp(mri, str(dest_test + "/mri/" + file_to_save))
                    save_bmp(mask, str(dest_test + "/mask/" + file_to_save))

                elif scan in val_scans:
                    save_bmp(mri, str(dest_val + "/mri/" + file_to_save))
                    save_bmp(mask, str(dest_val + "/mask/" + file_to_save))
