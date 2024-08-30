import pickle
import numpy as np
from get_data_path import get_data_path
# from rotate_and_translate import set_up_dataset_directory
import os
from PIL import Image
import random


def save_images(trans_mri, save_data_path, new_file_name):
    # Save file full paths
    mri_file_path = os.path.join(save_data_path, new_file_name)

    # Save images
    mri_img = Image.fromarray(trans_mri)
    mri_img.save(mri_file_path)
    return


def sample_between(values):
    # Unpack the list into the two values
    start, end = sorted(values)

    # Compute the center of the range
    center = (start + end) / 2

    # Generate a random sample around the center
    deviation = random.randint(-(end - start) // 2, (end - start) // 2)

    # Calculate the sampled value
    sampled_value = int(center + deviation)

    return sampled_value


def obtain_cropping_dict():
    """Returns a subject-specific cropping dict with information on how to crop that subject's scan
    The bounds for cropping were determined based on the subject's label. min_xy and min_z are the minimum number of
    pixels/slices that do not contain label that will be contained on all sides of the cropped image. Then, the bounds
    for cropping were randomly placed so that the label would fall within these constraints, and that they are
    normally-distributed about a centered scan.
    """

    with open("results/bounds_dict_HT.pkl", "rb") as f:
        bounds_dict = pickle.load(f)

    # xy_dim of cropped image
    xy_dim = 224
    z_dim = 56

    # Minimum pixels on either side of the box containing the label
    min_xy = 20
    min_z = 2

    # Iterate through each subject
    for subject in bounds_dict.keys():

        # Size of row and col size of the structure
        row_size = bounds_dict[subject]["row_size"]
        col_size = bounds_dict[subject]["col_size"]
        slice_size = bounds_dict[subject]["slice_size"]

        # Determine how many extra pixels we have to include that don't have label
        extra_rows = xy_dim - row_size
        extra_cols = xy_dim - col_size
        extra_slices = z_dim - slice_size

        # Extract indices of starting values
        row_inds = [bounds_dict[subject]["top_right"][0], bounds_dict[subject]["bottom_left"][0]]
        col_inds = [bounds_dict[subject]["bottom_left"][1], bounds_dict[subject]["top_right"][1]]
        slice_inds = bounds_dict[subject]["slice_bounds"]
        n_slices = slice_inds[1] - slice_inds[0] + 1

        # Determine range of potential starting bounds for the cropped image
        starting_row_bounds = [row_inds[1] + min_xy - xy_dim, row_inds[0] - min_xy]
        starting_col_bounds = [col_inds[1] + min_xy - xy_dim, col_inds[0] - min_xy]
        starting_slice_bounds = [slice_inds[1] + min_z - z_dim, slice_inds[0] - min_z]

        # Pick a random integer between these two bounds for each dimension
        row_start = sample_between(starting_row_bounds)
        col_start = sample_between(starting_col_bounds)
        slice_start = sample_between(starting_slice_bounds)

        if row_start < 0:
            row_start = 0
        if col_start < 0:
            col_start = 0
        if slice_start < 0:
            slice_start = 1

        # Define ending coordinate locations for each index
        row_end = row_start + xy_dim
        col_end = col_start + xy_dim
        slices_list = [slice_start + i for i in range(z_dim)]

        # Assemble coordinates for cropping in dictionary
        bounds_dict[subject]["row_start"] = row_start
        bounds_dict[subject]["row_end"] = row_end
        bounds_dict[subject]["col_start"] = col_start
        bounds_dict[subject]["col_end"] = col_end
        bounds_dict[subject]["slices_list"] = slices_list

    return bounds_dict

if __name__ == "__main__":

    random.seed(42)
    bounds_dict = obtain_cropping_dict()
    # Get current dataset we're cropping

    data_path = get_data_path("HT")
    dataset_types = ["test", "train", "val"]

    # Get saving dataset and setup
    save_data_path = get_data_path("CHT")

    for set_type in ["train", "test", "val"]:
        for img_type in ["mri", "mask"]:

            # iterate through each image and crop
            files = os.listdir(os.path.join(data_path, set_type, img_type))
            files.sort()

            for idx, file in enumerate(files):

                img_num = int(file.split("-")[1].split(".")[0])

                if img_num == 1:

                    subj_id = file.split("-")[0]

                    # Get cropping bounds for this subject
                    x_start = bounds_dict[subj_id]["col_start"]
                    x_end = bounds_dict[subj_id]["col_end"]
                    y_start = bounds_dict[subj_id]["row_start"]
                    y_end = bounds_dict[subj_id]["row_end"]
                    z_start = bounds_dict[subj_id]["slices_list"][0]
                    z_end = bounds_dict[subj_id]["slices_list"][-1]

                    # Iterate through included images
                    for idx, i in enumerate(range(z_start, z_end+1)):
                        open_file_name = subj_id + "-" + str(i).zfill(4) + ".bmp"
                        save_file_name = subj_id + "-" + str(idx+1).zfill(4) + ".bmp"

                        mri_name = os.path.join(data_path, set_type, img_type, open_file_name)

                        mri_img = Image.open(mri_name)
                        mri = np.array(mri_img)

                        if mri.shape == (512, 512, 3):
                            mri = mri[:, :, 0]

                        # Cropped image
                        mri_crop = mri[y_start:y_end, x_start:x_end]

                        # Save image
                        save_images(mri_crop, os.path.join(save_data_path, set_type, img_type), save_file_name)
                    print("Saved subject ", subj_id)
