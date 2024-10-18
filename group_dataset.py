from get_data_path import get_data_path
import os
import shutil

if __name__ == "__main__":

    # Define the base directories for train, test, and val
    data_folders = ['train', 'test', 'val']
    subfolders = ['mask', 'mri']  # Add other subfolders if necessary (e.g., 'mri')

    base_dir = get_data_path("HTO")

    for folder in data_folders:
        for subfolder in subfolders:
            subfolder_path = os.path.join(base_dir, folder, subfolder)

            # List all files in the current subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.bmp'):
                    subject_id = filename[:6]  # Get subject ID (first 6 characters)
                    subject_folder = os.path.join(subfolder_path, subject_id)

                    # Create the subject folder if it doesn't exist
                    if not os.path.exists(subject_folder):
                        os.makedirs(subject_folder)

                    # Move the file into the corresponding subject folder
                    src_file = os.path.join(subfolder_path, filename)
                    dst_file = os.path.join(subject_folder, filename)
                    shutil.move(src_file, dst_file)
                    print(f'Moved {filename} to {subject_folder}')

