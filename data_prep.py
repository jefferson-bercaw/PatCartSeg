import os
import numpy as np
from PIL import Image

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


def assemble_3d_mask(P_mask, PC_mask):
    mask = np.zeros((512, 512, 3))
    any_mask = P_mask | PC_mask

    mask[:, :, 0] = 1 - any_mask
    mask[:, :, 1] = P_mask
    mask[:, :, 2] = PC_mask

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
    print(scans)

    # Randomly select one subject for test and one subject for validation
    random_seed = 42
    np.random.seed(random_seed)

    test_subj = np.round(np.random.rand() * 7)
    val_subj = np.round(np.random.rand() * 7)
    while val_subj == test_subj:
        val_subj = np.round(np.random.rand() * 7)

    # Save these images in one combined folder
    train_scans = [entry["scan"] for entry in scans.values() if (entry['subject_num'] != test_subj and entry['subject_num'] != val_subj)]
    test_scans = [entry["scan"] for entry in scans.values() if entry['subject_num'] == test_subj]
    val_scans = [entry["scan"] for entry in scans.values() if entry['subject_num'] == val_subj]

    dest_train = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train_bmp"
    dest_test = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/test_bmp"
    dest_val = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/val_bmp"

    for scan in scans.keys():

        source_path = os.path.join(data_dir, scan)

        source_mri = os.path.join(source_path, "mri")
        source_P = os.path.join(source_path, "P")
        source_PC = os.path.join(source_path, "PC")

        # List of all image files
        image_files = get_bmp_files(source_mri)

        # Open files, assemble 3d masks, save mri and mask as npz files
        print(f"Saving for scan {scan}")

        for file in image_files:

            mri = read_bmp(source_mri, file) / 255.0
            P = read_bmp(source_P, file)
            PC = read_bmp(source_PC, file)

            mask = assemble_3d_mask(P, PC)

            file_to_save = file.split(".")[0] + ".bmp"

            # Save .npz files
            if save_opt:
                if scan in train_scans:
                    # np.savez(os.path.join(dest_train, "mri", file_to_save), arr=mri)
                    # np.savez(os.path.join(dest_train, "mask_3d", file_to_save), arr=mask)

                    save_bmp(mri, os.path.join(dest_train, "mri", file_to_save))
                    save_bmp(mask, os.path.join(dest_train, "mask_3d", file_to_save))

                elif scan in test_scans:
                    # np.savez(os.path.join(dest_test, "mri", file_to_save), arr=mri)
                    # np.savez(os.path.join(dest_test, "mask_3d", file_to_save), arr=mask)

                    save_bmp(mri, os.path.join(dest_test, "mri", file_to_save))
                    save_bmp(mask, os.path.join(dest_test, "mask_3d", file_to_save))

                elif scan in val_scans:
                    # np.savez(os.path.join(dest_val, "mri", file_to_save), arr=mri)
                    # np.savez(os.path.join(dest_val, "mask_3d", file_to_save), arr=mask)

                    save_bmp(mri, os.path.join(dest_val, "mri", file_to_save))
                    save_bmp(mask, os.path.join(dest_val, "mask_3d", file_to_save))

