import numpy as np
from PIL import Image
from evaluate import four_digit_number, assemble_mri_volume
import os
import matplotlib.pyplot as plt
import scipy
import pyvista as pv

# Workflow
# For each cartilage surface point, calculate nearest patellar point, calculate distance, store in location of patellar cartilage
# Plot heat map of cartilage thickness


def return_predicted_volumes(subj_name, model_name):
    cwd = os.getcwd()
    pred_folder = os.path.join(cwd, "results", model_name)

    p_pred_folder = os.path.join(pred_folder, "pat")
    pc_pred_folder = os.path.join(pred_folder, "pat_cart")

    # Catch models that do not have associated predictions yet
    if not os.path.exists(p_pred_folder) or not os.path.exists(pc_pred_folder):
        raise(ValueError, f"The prediction folder {p_pred_folder} or {pc_pred_folder} does not exist")

    image_list = [f"{subj_name}-{four_digit_number(i)}.bmp" for i in range(1, 120)]

    p_pred_names = [os.path.join(p_pred_folder, image_name) for image_name in image_list]
    pc_pred_names = [os.path.join(pc_pred_folder, image_name) for image_name in image_list]

    p_vol = assemble_mri_volume(p_pred_names)
    pc_vol = assemble_mri_volume(pc_pred_names)

    return p_vol, pc_vol


def return_pc_surface(pc_vol):
    pc_surf = np.zeros_like(pc_vol)

    for slice_num in range(pc_vol.shape[2]):
        pc_slice = pc_vol[:, :, slice_num]

        if np.max(pc_slice) > 0:
            pc_surf[:, :, slice_num] = get_outer_surface(pc_slice)
        else:
            pc_surf[:, :, slice_num] = pc_slice

    return pc_surf


def get_outer_surface(pc_slice):
    surf_slice = np.zeros_like(pc_slice)

    for i, row in enumerate(pc_slice):
        nonzero_inds = np.nonzero(row)[0]

        # If there are nonzero_inds in this row,
        if len(nonzero_inds) > 0:
            right_most_ind = nonzero_inds[-1]
            surf_slice[i, right_most_ind] = 1

    return surf_slice


def calculate_thickness(p_vol, pc_surf_mask):
    p_thick_map = np.zeros_like(pc_surf_mask)

    pc_inds = np.argwhere(pc_surf_mask)
    p_inds = np.argwhere(p_vol)

    distances = scipy.spatial.distance.cdist(pc_inds, p_inds)
    closest_indices = np.argmin(distances, axis=1)

    for i in range(len(pc_inds)):
        pc_coord = tuple(pc_inds[i])
        p_coord = tuple(p_inds[closest_indices[i]])
        dist = calculate_distance(p_coord, pc_coord)
        p_thick_map[p_coord] = dist

    return p_thick_map


def calculate_distance(p_coord, pc_coord):
    voxel_lengths = [0.3, 0.3, 1.0]  # voxel lengths in mm
    dist_comps = np.zeros(3)

    for i in range(len(p_coord)):
        dist_comps[i] = (p_coord[i] - pc_coord[i]) * voxel_lengths[i]

    dist = np.linalg.norm(dist_comps, ord=2)
    return dist


def remove_zero_slices(pc_thick_map):
    ind = 0
    num_non_zero_slices = np.nonzero(np.sum(pc_thick_map, axis=(0, 1)))[0].shape[0]
    pc_thick_trunc = np.zeros((pc_thick_map.shape[0], pc_thick_map.shape[1], num_non_zero_slices))

    for slice_num in range(pc_thick_map.shape[2]):
        pc_slice = pc_thick_map[:, :, slice_num]
        if np.sum(pc_slice) > 0:
            pc_thick_trunc[:, :, ind] = pc_slice
            ind += 1

    return pc_thick_trunc


def visualize_thickness_map(p_thick_map):

    p_thick_trunc = remove_zero_slices(p_thick_map)

    cmap = plt.colormaps['viridis']
    norm = plt.Normalize(p_thick_trunc.min(), p_thick_trunc.max())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x in range(p_thick_trunc.shape[0]):
        for y in range(p_thick_trunc.shape[1]):
            for z in range(p_thick_trunc.shape[2]):
                thickness = p_thick_trunc[x, y, z]
                if thickness > 0:  # Only plot non-zero thickness values
                    color = cmap(norm(thickness))  # Map thickness value to color
                    ax.voxels(x, y, z, filled=p_thick_trunc > 0, color=color, edgecolor='k')  # Plot the voxel

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(p_thick_trunc)
    plt.colorbar(sm, label='Thickness')

    plt.show()
    return


if __name__ == '__main__':

    subj_name = "AS_018"
    model_name = "unet_2024-04-17_08-06-28"

    # Load in patella and patellar cartilage volumes
    p_vol, pc_vol = return_predicted_volumes(subj_name, model_name)

    # Post-processing: Fill holes, remove stray pixels, in both volumes

    # Edit mask to get the right most pixels for the patellar cartilage
    pc_surf_mask = return_pc_surface(pc_vol)

    # Visualize
    # for slice_num in range(pc_surf_mask.shape[2]):
    #     pc_surf_slice = pc_surf_mask[:, :, slice_num]
    #     if np.max(pc_surf_slice) > 0:
    #         plt.imshow(pc_surf_slice, cmap='gray')
    #         plt.show()
    #         plt.close()

    p_thick_map = calculate_thickness(p_vol, pc_surf_mask)

    visualize_thickness_map(p_thick_map)

    # For each cartilage surface point, calculate nearest patellar point, calculate distance, store in location of patellar cartilage


    print("HEHE")
