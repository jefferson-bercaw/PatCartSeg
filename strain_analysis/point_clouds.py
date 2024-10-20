import numpy as np
from evaluate import four_digit_number, assemble_mri_volume
import os
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage
import pyvista as pv
import pickle
import open3d as o3d
from get_data_path import get_data_path


def return_predicted_volumes(subj_name, model_name):
    cwd = os.getcwd()
    pred_folder = os.path.join(cwd, "../results", model_name)

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

    p_vol[p_vol > 0] = 1
    pc_vol[pc_vol > 0] = 1

    return p_vol, pc_vol


def return_p_surface(p_vol):
    """Takes in a (256, 256, 120) binary mask for patella and returns (256, 256, 120) for patella surface"""
    p_surf = np.zeros_like(p_vol)

    for slice_num in range(p_vol.shape[2]):
        p_slice = p_vol[:, :, slice_num]

        if np.max(p_slice) > 0:
            p_surf[:, :, slice_num] = get_entire_surface(p_slice)

            # Visualize
            # plt.imshow(p_surf[:, :, slice_num])
            # plt.show()

    return p_surf


def get_entire_surface(p_slice):
    """Takes in binary slice for entire patella and returns binary slice for just the surface"""
    # Approach: Get max and min pixel locations for each column and row, and find the union of this
    boundary_locs = list()
    p_surf = np.zeros_like(p_slice)

    for i, row in enumerate(p_slice):  # iterate through each row
        nonzero_inds = np.nonzero(row)[0]
        if len(nonzero_inds) > 0:
            left_most_ind = nonzero_inds[0]
            right_most_ind = nonzero_inds[-1]
            boundary_locs.append([i, left_most_ind])  # list of row, col indices of surface pixels
            boundary_locs.append([i, right_most_ind])  # list of row, col indices of surface pixels

    for col in range(p_slice.shape[1]):  # iterate through each col
        column = p_slice[:, col]
        nonzero_inds = np.nonzero(column)[0]

        if len(nonzero_inds) > 0:
            top_most_ind = nonzero_inds[0]
            bottom_most_ind = nonzero_inds[-1]
            boundary_locs.append([top_most_ind, col])  # list of row, col indices of surface pixels
            boundary_locs.append([bottom_most_ind, col])  # list of row, col indices of surface pixels

    for row, col in boundary_locs:
        p_surf[row, col] = 1

    # plt.imshow(p_surf)
    # plt.show()

    return p_surf


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


def get_patella_point_cloud(p_vol):
    """Takes in a volume of the patella surface (256, 256, 120) and returns a (nx3) point cloud (interp)"""
    voxel_lengths = [0.3, 0.3, 1.0]  # voxel lengths in mm
    p_inds = np.argwhere(p_vol)
    p_pos = p_inds * voxel_lengths
    # p_pos = interpolate_patella(p_pos)
    return p_pos


def calculate_thickness(p_pos, pc_pos):
    """Takes in a patella binary volume and a patellar cartilage surface binary volume, and returns the same patellar
     cartilage volume, with cartilage thickness values in the location of the patellar cartilage surface"""

    # voxel_lengths = [0.3, 0.3, 1.0]  # voxel lengths in mm

    # Find the indices of the patella and patellar cartilage
    # pc_inds = np.argwhere(pc_surf_mask)
    # p_inds = np.argwhere(p_vol)
    #
    # # Find the x, y, and z coordinates of the patella and patellar cartilage
    # pc_pos = pc_inds * voxel_lengths
    # p_pos = p_inds * voxel_lengths

    # Add points to the patella along the surface for a finer mesh
    # p_pos = interpolate_patella(p_pos)

    # Calculate distance matrix between pc and p, and find the closest patella point for each patellar cartilage point
    distances = scipy.spatial.distance.cdist(pc_pos, p_pos)
    closest_indices = np.argmin(distances, axis=1)

    pc_coords_array = np.zeros((len(pc_pos), 4))

    for i in range(len(pc_pos)):
        # Get x, y, z coordinates of the PC point and the nearest patella point
        pc_coord = pc_pos[i]
        p_coord = p_pos[closest_indices[i]]

        # Calculate the distance between the two
        dist = np.linalg.norm(p_coord-pc_coord)

        # Store in thickness map (256, 256, 120)
        # pc_ind_here = pc_inds[i]
        pc_coords_array[i, :] = [pc_coord[0], pc_coord[1], pc_coord[2], dist]

    return pc_coords_array


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


def organize_coordinate_array(pc_thick_map):
    """Turns (xy_dim, xy_dim, n_slices) ndarray into (n, 4) array of [x, y, z, thick] values for each PC coord on PC"""
    inds = np.nonzero(pc_thick_map)
    voxel_lengths = [0.3, 0.3, 1.0]  # voxel lengths in mm

    coords_array = np.zeros((len(inds[0]), 4))
    for i in range(len(inds[0])):
        x, y, z = inds[0][i], inds[1][i], inds[2][i]
        value = pc_thick_map[x, y, z]
        coords_array[i] = [x * voxel_lengths[0], y * voxel_lengths[1], z * voxel_lengths[2], value]
    return coords_array


def upsample_pc_coords_array(coords_array):
    """Takes in patellar cartilage coords and thickness array (nx4), creates a surface using delaunay_2d, subdivides
     this surface by a factor of 2, and returns the interpolated coords and thickness point clouds."""
    point_cloud = pv.PolyData(np.transpose([coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]]))
    point_cloud['Cart. Thickness (mm)'] = coords_array[:, 3]

    surf = point_cloud.delaunay_2d()
    surf['Cart. Thickness (mm)'] = coords_array[:, 3]

    # Upsample cartilage surface
    surface_upsampled = surf.subdivide(nsub=2)
    xyz_coords = surface_upsampled.points
    thickness = surface_upsampled.point_data["Cart. Thickness (mm)"]
    coords_array_upsampled = np.concatenate((xyz_coords, thickness[:, np.newaxis]), axis=1)
    return coords_array_upsampled


def visualize_thickness_map(coords_array):
    # coords_array = organize_coordinate_array(pc_thick_map)
    point_cloud = pv.PolyData(np.transpose([coords_array[:, 0], coords_array[:, 1], coords_array[:, 2]]))
    point_cloud['Cart. Thickness (mm)'] = coords_array[:, 3]

    surf = point_cloud.delaunay_2d()
    surf['Cart. Thickness (mm)'] = coords_array[:, 3]

    # Upsample cartilage surface
    surface_upsampled = surf.subdivide(nsub=2)

    # Plot 3d triangle-ized surface
    surface_upsampled.plot(show_edges=False)
    return
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


def interpolate_patella(p_pos):
    """Takes in nx3 ndarray of patella positions, and interpolates them"""
    p_point_cloud = pv.PolyData(np.transpose([p_pos[:, 0], p_pos[:, 1], p_pos[:, 2]]))

    # Construct patella surface and resample
    p_surf = p_point_cloud.delaunay_2d().poisson_surface()
    p_surf = p_surf.subdivide(nsub=1)
    p_surf.plot(show_edges=True)

    # Get new points
    p_pos = p_surf.points
    return p_pos


def store_point_clouds(point_clouds, p_coords_array, pc_coords_array, p_right_coords_array, subj_name):
    """Stores point clouds in a dictionary to be dumped"""
    point_clouds[subj_name] = {}
    point_clouds[subj_name]["p_coords_array"] = p_coords_array
    point_clouds[subj_name]["pc_coords_array"] = pc_coords_array
    point_clouds[subj_name]["p_right_coords_array"] = p_right_coords_array
    return point_clouds


def remove_nocart_slices(p_vol, pc_vol):
    """Removes slices in patella volume with no cartilage associated"""
    for slice_num in range(p_vol.shape[2]):
        p_slice = p_vol[:, :, slice_num]
        pc_slice = pc_vol[:, :, slice_num]

        if np.max(p_slice) > 0 and np.max(pc_slice) == 0:  # if there is patella here but no cartilage
            p_vol[:, :, slice_num] = 0  # replace all pixels in this slice with zeros
    return p_vol


def remove_patella_outliers(p_vol):
    """Removes disconnected pixels from the patella"""
    # Iterate through each slice (3rd dimension)
    # Calculate 2d connectivity of each pixel in the binary mask
    # Remove pixels that stand alone (have no connectivity), or pixels that are only connected diagonally

    # Step 2: Not doing:
    # Find the middle slice of the patella
    # Calculate centroid of the patella of this slice and the max distance of a True pixel from this centroid
    # Iterate through each slice (3rd dimension)
    # Calculate distance from the centroid of the middle slice
    # If larger than 1.5x the max distance, set to False

    # Step 1: Remove disconnected pixels and those connected only diagonally in each 2D slice
    cleaned_vol = np.zeros_like(p_vol, dtype=bool)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])  # 4-connectivity structure

    for i in range(p_vol.shape[2]):
        slice_ = p_vol[:, :, i]
        labeled_slice, num_features = ndimage.label(slice_, structure=structure)

        # Filter out small components
        sizes = ndimage.sum(slice_, labeled_slice, range(1, num_features + 1))

        for j in range(1, num_features + 1):
            if sizes[j - 1] >= 600:
                cleaned_vol[:, :, i][labeled_slice == j] = True

    return cleaned_vol


def extract_right_patellar_volume(p_vol, pc_vol):
    """Extracts the patellar pixels at the patellar cartilage interface"""
    p_right_vol = np.zeros_like(p_vol)

    for slice_num in range(p_vol.shape[2]):
        p_slice = p_vol[:, :, slice_num]
        pc_slice = pc_vol[:, :, slice_num]

        if np.max(p_slice) > 0 and np.max(pc_slice) > 0:  # if there are patella amd patellar cartilage slices
            pc_true_inds = np.argwhere(pc_slice)

            # Iterate through every true cartilage pixel
            for row, col in pc_true_inds:

                # Create distance map of all patella points from row, col
                row_ind, col_ind = np.indices(p_slice.shape)  # indices of patella
                distances = np.sqrt((row_ind - row) ** 2 + (col_ind - col) ** 2)
                dist_map = np.zeros_like(pc_slice)
                dist_map[p_slice > 0] = distances[p_slice > 0]

                # Find the row and column location of the minimum non_zero distance
                masked_distances = np.ma.masked_equal(dist_map, 0)
                min_dist = masked_distances.min()
                row_P, col_P = np.where(dist_map == min_dist)

                # Assign true to the row_P and col_P location
                p_right_vol[row_P, col_P, slice_num] = 1

            # plt.imshow(p_right_vol[:, :, slice_num])
            # plt.show()
            # print("Slice")

    return p_right_vol


def return_true_volumes(subj_name):
    """Returns true volumes from manual segmentation"""
    data_path = get_data_path("cHT")

    true_folder = os.path.join(data_path, "test", "mask")

    image_list = [f"{subj_name}-{four_digit_number(i)}.bmp" for i in range(1, 120)]

    true_names = [os.path.join(true_folder, image_name) for image_name in image_list]

    true_vol = assemble_mri_volume(true_names, xy_dim=256)

    p_vol = np.zeros_like(true_vol)
    pc_vol = np.zeros_like(true_vol)

    inds_p = np.logical_and(0.003 < true_vol, true_vol < 0.006)
    inds_pc = true_vol > 0.006

    p_vol[inds_p] = 1
    pc_vol[inds_pc] = 1

    return p_vol, pc_vol


def get_coordinate_arrays(p_vol, pc_vol):
    """Transforms (256, 256, 120) ndarrays for the patella and patellar cartilage predictions to (n, 3) ndarray for
    the patella, (n, 3) for the cartilage and cartilage thickness, and (n, 3) for the articulating surface of the
    patella"""
    # Zero all slices of the patella that do not have patellar cartilage
    # p_vol = remove_nocart_slices(p_vol, pc_vol)

    # Remove outliers based on distance from centroid and connectivity
    p_vol = remove_patella_outliers(p_vol)

    # Get right patellar volume (at the cartilage interface)
    p_right_vol = extract_right_patellar_volume(p_vol, pc_vol)

    # Edit mask to get the surface pixels (no middle pixels) for the patella and get the point cloud
    p_surf_mask = return_p_surface(p_vol)
    p_coords_array = get_patella_point_cloud(p_surf_mask)

    # Interpolate the patella surface
    # p_interp_coords_array = interpolate_patella(p_raw_coords_array)

    # Edit mask to get the right most pixels for the patellar cartilage and patella surf
    pc_surf_mask = return_pc_surface(pc_vol)
    pc_coords_array = get_patella_point_cloud(pc_surf_mask)

    # For each cartilage surface pt, calculate nearest P pt, calculate dist, store val in PC coord
    # pc_coords_array = calculate_thickness(p_surf_mask, pc_surf_mask)

    return p_coords_array, pc_coords_array


def export_point_cloud(subj_name, tissue_type, point_cloud):
    """Export a nx3 point cloud to a .txt file"""
    np.savetxt(f"./geomagic/{subj_name}_{tissue_type}.txt", point_cloud, delimiter='\t', fmt='%.6f')
    return


def import_point_cloud(subj_name, tissue_type):
    """Imports a nx3 point cloud from a .pcd file"""
    filename = f"./geomagic/{subj_name}_{tissue_type}.pcd"
    ptcld = o3d.io.read_point_cloud(filename)
    coord_array = np.asarray(ptcld.points)
    return coord_array


if __name__ == '__main__':

    # Specify model name and subject name(s)
    # subj_names = ["AS_018", "AS_019", "AS_020", "AS_021", "AS_022", "AS_023"]
    # model_name = "unet_2024-04-17_08-06-28"
    subj_names = ["AS_006", "AS_007", "AS_008", "AS_009", "AS_010", "AS_011"]
    model_name = "unet_2024-06-16_16-41-22_cHT5"

    thickness_values = list()

    # Initialize dictionary
    point_clouds = {}

    # Thickness Loop
    for subj_name in subj_names:
        print(f"Subject {subj_name}")

        # Load in patella and patellar cartilage volumes for predicted and true scans
        p_vol, pc_vol = return_predicted_volumes(subj_name, model_name)
        # p_vol_true, pc_vol_true = return_true_volumes(subj_name)

        # # Get coordinate arrays for predicted and true scans
        p_coords_array, pc_coords_array, p_right_coords_array = get_coordinate_arrays(p_vol, pc_vol)
        # p_true_coords_array, pc_true_coords_array, pc_right_coords_array = get_coordinate_arrays(p_vol_true, pc_vol_true)

        # # Export to point clouds for Geomagic to smooth/interpolate
        export_point_cloud(subj_name, "P", p_coords_array)
        export_point_cloud(subj_name, "PC", pc_coords_array)

        # Load in new point clouds
        p_coords_array = import_point_cloud(subj_name, "P")
        pc_coords_array = import_point_cloud(subj_name, "PC")

        # Register the patellae together
        # p_points_moved, transform = move_patella(p_coords_array, p_true_coords_array, True)  # True moving to predicted

        # Calculate cartilage thickness maps
        pc_thick_map = calculate_thickness(p_coords_array, pc_coords_array)
        print(f"Mean thickness: {np.mean(pc_thick_map[:,3])}")

        # Save (nx4) cartilage thickness maps

        ### OLD
        # # Calculate coord array and store thickness values for this scan
        # pc_coords_array = organize_coordinate_array(pc_thick_map)

        # Upsample the cartilage point cloud
        # pc_coords_array = upsample_pc_coords_array(pc_coords_array)

        # Store point clouds in a dictionary
        # point_clouds = store_point_clouds(point_clouds, p_coords_array, pc_coords_array, p_right_coords_array, subj_name)

        # Store thickness values for distribution analysis
        # thickness_values.append(pc_coords_array[:, 3])

        # Visualize the map
        visualize_thickness_map(pc_thick_map)
        print("Done")
    # plot_thickness_distributions(thickness_values, model_name)

    with open("point_clouds.pkl", 'wb') as f:
        pickle.dump(point_clouds, f)
