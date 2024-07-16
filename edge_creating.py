import os
from get_data_path import get_data_path
from PIL import Image
from rotate_and_translate import set_up_dataset_directory
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt


def get_outer_surface(mask_slice):
    """Binary of the outermost surface of this slice (right most surface)"""
    surf_slice = np.zeros_like(mask_slice)

    for i, row in enumerate(mask_slice):
        nonzero_inds = np.nonzero(row)[0]

        # If there are nonzero_inds in this row,
        if len(nonzero_inds) > 0:
            right_most_ind = nonzero_inds[-1]
            surf_slice[i, right_most_ind] = 1

    return surf_slice


def split_mask(mask):
    """Splits 0, 1, and 2 into p and pc masks"""
    p = np.zeros_like(mask)
    pc = np.zeros_like(mask)

    p[mask == 1] = 1
    pc[mask == 2] = 1

    return p, pc


def assemble_mask(p, pc, p_surf, pc_surf):
    """Assembles mask of p and pc with outer surface, where
    p = 1
    pc = 2
    p_surf = 3
    pc_surf = 4
    """
    mask = np.zeros_like(p)

    mask[p == 1] = 1
    mask[pc == 1] = 2
    mask[p_surf == 1] = 3
    mask[pc_surf == 1] = 4

    return mask


if __name__ == "__main__":
    # Starting data path
    data_path = get_data_path("ctHT")
    save_data_path = get_data_path("cteHT")

    set_up_dataset_directory(save_data_path)

    # image locations
    sets = ["train", "val", "test"]
    for set in sets:
        mri_path = os.path.join(data_path, set, "mri")
        mask_path = os.path.join(data_path, set, "mask")

        mri_dest = os.path.join(save_data_path, set, "mri")
        mask_dest = os.path.join(save_data_path, set, "mask")

        # list all filenames in this set
        mask_files = os.listdir(mask_path)

        for mask_file in mask_files:
            mask_name = os.path.join(mask_path, mask_file)

            mask = Image.open(mask_name)
            mask = np.asarray(mask)

            p, pc = split_mask(mask)

            p_surf = get_outer_surface(p)
            pc_surf = get_outer_surface(pc)

            mask_out = assemble_mask(p, pc, p_surf, pc_surf)

            # plt.imshow(mask_out * 50)
            # plt.show()

            # Save mask
            mask_out = Image.fromarray(mask_out)
            mask_out.save(os.path.join(mask_dest, mask_file))

            # Move MRI
            mri_file = os.path.join(mri_path, mask_file)
            mri_dest_file = os.path.join(mri_dest, mask_file)
            copyfile(mri_file, mri_dest_file)

