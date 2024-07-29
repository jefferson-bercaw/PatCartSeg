import os
import random
import shutil
import pandas as pd
from get_data_path import get_data_path


def search_and_randomize_folders(directory):
    # Step 1: Search for all folders in the directory
    folder_names = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Step 2: Randomly shuffle the folder names
    shuffled_folders = folder_names[:]
    random.shuffle(shuffled_folders)

    # Step 3: Create a corresponding list of indices
    indices = list(range(len(folder_names)))

    return indices, shuffled_folders


def write_random_excel(directory, folder_names, shuffled_folders):
    # Step 1: Create a DataFrame with the original and shuffled folder names
    df = pd.DataFrame({'Original Folder Names': shuffled_folders, 'Randomized Folder Names': folder_names})

    # Step 2: Write the DataFrame to an Excel file
    df.to_excel(os.path.join(directory, 'random_info.xlsx'), index=False)


def copy_folders_to_new_directory(original_directory, new_directory, folder_names, shuffled_folders):
    # Make sure the destination directory exists
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Copy all contents from src to dst
    for original_name, randomized_name in zip(folder_names, shuffled_folders):
        src = os.path.join(original_directory, original_name)
        dst = os.path.join(new_directory, randomized_name)

        # Make sure the destination directory exists
        if not os.path.exists(dst):
            os.makedirs(dst)

        # Copy all contents from src to dst
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        print(f"Copied {original_name} to {randomized_name}")

def get_random_info(directory):
    # Load in excel doc
    df = pd.read_excel(os.path.join(directory, 'random_info.xlsx'))

    # Split two columns and read in as lists
    shuffled_folders = df['Original Folder Names'].tolist()
    numbered_folders = df['Randomized Folder Names'].tolist()

    return numbered_folders, shuffled_folders


if __name__ == "__main__":
    # Create Randomization #
    start_path = get_data_path("Paranjape_Predictions")
    model_name = "unet_2024-07-11_00-40-25_ctHT5"

    directory = os.path.join(start_path, model_name)
    original_directory = os.path.join(start_path, model_name, "original")
    randomized_directory = os.path.join(start_path, model_name, "randomized")

    # Randomize shuffled folders
    numbered_folders, shuffled_folders = search_and_randomize_folders(original_directory)

    # Write excel spreadsheet containing randomized info
    write_random_excel(directory, numbered_folders, shuffled_folders)

    # Copy folders to new directory with randomized folder names
    copy_folders_to_new_directory(original_directory, randomized_directory, numbered_folders, shuffled_folders)

    # Go back to non-randomized folder names #
    # Load in the excel file
    numbered_folders, shuffled_folders = get_random_info(directory)

    # Copy folders to new directory with randomized folder names
    copy_folders_to_new_directory(randomized_directory, original_directory, numbered_folders, shuffled_folders)