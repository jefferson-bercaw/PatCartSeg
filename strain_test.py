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

    p_vol = assemble_mri_volume(p_pred_names, xy_dim=256)
    pc_vol = assemble_mri_volume(pc_pred_names, xy_dim=256)

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
    voxel_lengths = [0.3, 0.3, 1.0]  # voxel lengths in mm

    pc_thick_map = np.zeros_like(pc_surf_mask)

    pc_inds = np.argwhere(pc_surf_mask)
    p_inds = np.argwhere(p_vol)

    pc_pos = pc_inds * voxel_lengths
    p_pos = p_inds * voxel_lengths

    distances = scipy.spatial.distance.cdist(pc_pos, p_pos)
    closest_indices = np.argmin(distances, axis=1)

    for i in range(len(pc_inds)):
        pc_coord = tuple(pc_inds[i])
        p_coord = tuple(p_inds[closest_indices[i]])
        dist = calculate_distance(p_coord, pc_coord)
        pc_thick_map[pc_coord] = dist

    return pc_thick_map


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


def organize_coordinate_array(p_thick_map):
    """Turns (xy_dim, xy_dim, n_slices) ndarray into (n, 4) array of [x, y, z, thick] values for each PC coord on P"""
    inds = np.nonzero(p_thick_map)
    voxel_lengths = [0.3, 0.3, 1.0]  # voxel lengths in mm

    coords_array = np.zeros((len(inds[0]), 4))
    for i in range(len(inds[0])):
        x, y, z = inds[0][i], inds[1][i], inds[2][i]
        value = p_thick_map[x, y, z]
        coords_array[i] = [x * voxel_lengths[0], y * voxel_lengths[1], z * voxel_lengths[2], value]
    return coords_array


def visualize_thickness_map(p_thick_map):
    coords_array = organize_coordinate_array(p_thick_map)
    point_cloud = pv.PolyData(np.transpose([coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]]))
    point_cloud['Cart. Thickness (mm)'] = coords_array[:, 3]

    surf = point_cloud.delaunay_2d()
    surf['Cart. Thickness (mm)'] = coords_array[:, 3]

    # Plot 3d triangle-ized surface
    surf.plot(show_edges=True, line_width=0.2)

    # Plot point cloud
    # plotter = pv.Plotter()
    # plotter.add_points(point_cloud, point_size=20)
    # plotter.show()


def plot_thickness_distributions(thickness_values, model_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.boxplot(thickness_values)
    ax.set_xticklabels(["Pre1", "Post 3mi", "Rec1", "Pre2", "Post 10mi", "Rec2"])
    ax.set_ylabel("Patellar Cartilage Thickness (mm)")
    ax.set_title(f"{model_name}: PC Thickness without Post-Processing")
    plt.show()
    return


if __name__ == '__main__':

    # Specify model name and subject name(s)
    # subj_names = ["AS_018", "AS_019", "AS_020", "AS_021", "AS_022", "AS_023"]
    # model_name = "unet_2024-04-17_08-06-28"

    subj_names = ["AS_006", "AS_007", "AS_008", "AS_009", "AS_010", "AS_011"]
    model_name = "unet_2024-05-15_07-23-17_cHT5"

    thickness_values = list()

    for subj_name in subj_names:

        # Load in patella and patellar cartilage volumes
        p_vol, pc_vol = return_predicted_volumes(subj_name, model_name)

        # Post-processing: Fill holes, remove stray pixels, in both volumes

        # Edit mask to get the right most pixels for the patellar cartilage
        pc_surf_mask = return_pc_surface(pc_vol)
        num_cart_points = np.sum(pc_surf_mask)

        # for slice_num in range(pc_surf_mask.shape[2]):
        #     pc_surf_slice = pc_surf_mask[:, :, slice_num]
        #     if np.max(pc_surf_slice) > 0:
        #         plt.imshow(pc_surf_slice, cmap='gray')
        #         plt.show()
        #         plt.close()

        # For each cartilage surface point, calculate nearest patellar point, calculate distance, store in patella coord.
        p_thick_map = calculate_thickness(p_vol, pc_surf_mask)

        # Calculate coord array and store thickness values for this scan
        coords_array = organize_coordinate_array(p_thick_map)
        num_cart_vox = len(coords_array)

        thickness_values.append(coords_array[:, 3])

        # Visualize the map
        visualize_thickness_map(p_thick_map)

    plot_thickness_distributions(thickness_values, model_name)