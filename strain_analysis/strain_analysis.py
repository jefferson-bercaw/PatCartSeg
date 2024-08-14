import numpy as np
import open3d as o3d
import pickle
import os
import copy
import scipy
import pyvista as pv
from RANSACReg import preprocess_point_cloud, execute_global_registration, refine_registration
import matplotlib.pyplot as plt
from registration import move_patella
from get_data_path import get_data_path


def load_point_cloud():
    with open(os.path.join(os.getcwd(), "point_clouds.pkl"), 'rb') as f:
        point_clouds = pickle.load(f)
    return point_clouds


def get_patella_ptclds(point_clouds, subj_name):
    # Get patella point cloud from point cloud dictionary
    p_points = point_clouds[subj_name]["p_coords_array"]
    p_right_points = point_clouds[subj_name]["p_right_coords_array"]

    p_ptcld = o3d.geometry.PointCloud()
    p_ptcld.points = o3d.utility.Vector3dVector(p_points)

    p_right_ptcld = o3d.geometry.PointCloud()
    p_right_ptcld.points = o3d.utility.Vector3dVector(p_right_points)

    return p_ptcld, p_right_ptcld


def get_cartilage_ptcld(point_clouds, subj_name):
    # Get patella point cloud from point cloud dictionary
    pc_points = point_clouds[subj_name]["pc_coords_array"][:, 0:3]
    thickness = point_clouds[subj_name]["pc_coords_array"][:, 3]
    pc_ptcld = o3d.geometry.PointCloud()
    pc_ptcld.points = o3d.utility.Vector3dVector(pc_points)
    return pc_ptcld, thickness


def get_transformations(fixed_p_ptcld, moving_p_ptcld):
    voxel_size = 2

    fixed_p_ptcld, fixed_fpfh = preprocess_point_cloud(fixed_p_ptcld, voxel_size)
    moving_p_ptcld, moving_fpfh = preprocess_point_cloud(moving_p_ptcld, voxel_size)

    result_ransac = execute_global_registration(moving_p_ptcld, fixed_p_ptcld,
                                                moving_fpfh, fixed_fpfh, voxel_size)

    result_icp = refine_registration(moving_p_ptcld, fixed_p_ptcld, moving_fpfh, fixed_fpfh, voxel_size, result_ransac)

    return result_ransac, result_icp


def store_transformations(transformations, subj_name, result_icp):
    transformations[subj_name] = {}
    transformations[subj_name]["icp"] = result_icp
    return transformations


def average_thickness_values(pc_ptcld, thickness):
    # Iterate through every point
    # Calculate indices of nearest points
    # Average thickness values at that point and reassign this thickness value

    pc_points = np.asarray(pc_ptcld.points)
    radius_mm = 2.5  # radius of search
    thickness_new = np.zeros_like(thickness)

    for idx, coord in enumerate(pc_points):
        distances = np.linalg.norm(coord - pc_points, axis=1)
        inds = distances < radius_mm
        thick_to_avg = thickness[inds]

        # plt.hist(thick_to_avg)
        # plt.show()

        thickness_new[idx] = np.mean(thick_to_avg)

    return pc_points, thickness_new


def remove_outer_boundaries(pc_ptcld, thickness, radius, n):
    for i in range(n):
        # Estimate normals for the entire point cloud
        pc_ptcld.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

        # Compute point density
        kdtree = o3d.geometry.KDTreeFlann(pc_ptcld)
        densities = np.zeros(len(pc_ptcld.points))

        for i, point in enumerate(pc_ptcld.points):
            [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
            densities[i] = len(idx)
        points = np.asarray(pc_ptcld.points)

        # plt.hist(densities, bins=15)
        # plt.show()
        # Threshold for edge removal
        # density_threshold = round(np.max(densities) / 2)  # cropping_2
        density_threshold = np.percentile(densities, 5)  # cropping_3 (duration) and cropping_4 (speed)
        # Filter points based on density
        filtered_points = []
        filtered_thickness = []
        filtered_points = points[densities >= density_threshold]
        filtered_thickness = thickness[densities >= density_threshold]

        # for i, density in enumerate(densities):
        #     if density >= density_threshold:
        #         filtered_points.append(pc_ptcld.points[i])
        #         filtered_thickness.append(thickness[i])

        # Create a new point cloud from the filtered points
        pc_ptcld = o3d.geometry.PointCloud()
        pc_ptcld.points = o3d.utility.Vector3dVector(np.array(filtered_points))

        thickness = np.array(filtered_thickness)

    return pc_ptcld, thickness


def produce_strain_map(pre_pc_ptcld, pre_thickness, post_pc_ptcld, post_thickness, output, save_location, pre_scan):

    radius_mm = 2.5
    pre_pc_points, pre_thickness = average_thickness_values(pre_pc_ptcld, pre_thickness)
    post_pc_points, post_thickness = average_thickness_values(post_pc_ptcld, post_thickness)

    # Visualize thickness maps
    thick_pre_map = np.concatenate((np.asarray(pre_pc_points), pre_thickness[:, np.newaxis]), axis=1)
    thick_post_map = np.concatenate((np.asarray(post_pc_points), post_thickness[:, np.newaxis]), axis=1)

    # Save thickness maps averaged
    plot_thickness_array(thick_pre_map, pre_scan, save_location, "thick_avg", "Thickness (mm)")
    plot_thickness_array(thick_post_map, pre_scan[:11]+'post', save_location, "thick_avg", "Thickness (mm)")

    moving_pc = thick_post_map[:, 0:3]
    fixed_pc = thick_pre_map[:, 0:3]

    # compute distances
    distances = scipy.spatial.distance.cdist(moving_pc, fixed_pc)
    closest_indices = np.argmin(distances, axis=1)  # indices of the fixed_pc closest to the moving_pc

    # Initialize arrays
    avg_coord = []
    strain = []

    # Iterate through the moving_pc (post)
    for i in range(len(moving_pc)):
        #  Get x, y, z coordinates of the moving and fixed PC point
        moving_coord = moving_pc[i]  # post coord
        fixed_coord = fixed_pc[closest_indices[i]]  # pre coord

        # Find average thickness value within 2.5 mm radius
        moving_dists = np.linalg.norm(moving_coord - moving_pc, axis=1)
        # moving_inds = moving_dists < radius_mm
        # moving_thick = np.mean(post_thickness[moving_inds])
        moving_thick = post_thickness[i]

        fixed_dists = np.linalg.norm(fixed_coord - fixed_pc, axis=1)
        # fixed_inds = fixed_dists < radius_mm
        # fixed_thick = np.mean(pre_thickness[fixed_inds])
        fixed_thick = pre_thickness[closest_indices[i]]

        # Threshold distance. If the distance between these two coordinates isn't too large, add to strain map
        dist_thresh = 1  # distance [mm] that signifies a "good" comparison
        if np.linalg.norm(moving_coord - fixed_coord) < dist_thresh:
            pre_thick = fixed_thick
            post_thick = moving_thick

            avg_coord.append(moving_coord)  # fixed (pre) coord

            strain_here = (post_thick - pre_thick) / pre_thick
            strain.append(strain_here)

    # Convert coords and strain lists to numpy arrays
    avg_coord = np.array(avg_coord)
    strain_original = np.array(strain)

    # Create nx4 strain ndarray
    strain_original_map = np.concatenate((avg_coord, strain_original[:, np.newaxis]), axis=1)
    plot_thickness_array(strain_original_map, pre_scan[:11], save_location, "strain_raw", "Strain")

    # Average strain map values within 2.5 mm radius
    strain_ptcld_original = o3d.geometry.PointCloud()
    strain_ptcld_original.points = o3d.utility.Vector3dVector(avg_coord)
    strain_pc_points, strain = average_thickness_values(strain_ptcld_original, strain_original)

    # Create nx4 strain ndarray
    strain_original_map = np.concatenate((np.asarray(strain_pc_points), strain[:, np.newaxis]), axis=1)
    plot_thickness_array(strain_original_map, pre_scan[:11], save_location, "strain_avg", "Strain")

    strain_ptcld_avg = o3d.geometry.PointCloud()
    strain_ptcld_avg.points = o3d.utility.Vector3dVector(strain_pc_points)
    strain_ptcld_final, strain_final = remove_outer_boundaries(strain_ptcld_avg, strain, radius=5.0, n=8)

    # Create nx4 boundaries removed strain array
    strain_final_map = np.concatenate((np.asarray(strain_ptcld_final.points), strain_final[:, np.newaxis]), axis=1)
    plot_thickness_array(strain_final_map, pre_scan[:11], save_location, "strain_boundaries_removed", "Strain")

    # Perform parametric analysis on averaged, but cropped to one set of parameters, strain map
    cropping_list = parameterize_cropping(strain_ptcld_avg, strain)

    strain_map = np.concatenate((np.asarray(strain_ptcld_final.points), strain_final[:, np.newaxis]), axis=1)

    # save_strain_map(strain_map, "strain", f"R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Test_Lauren\\Manual_Segmentations\\Strain\\{subj}\\{dist}mi_strain_map_{idx}.pdf")

    # strain_ptcld = o3d.geometry.PointCloud()
    # strain_ptcld.points = o3d.utility.Vector3dVector(strain_map[:, 0:3])

    # plt.plot(iteration, mean_strain)
    # plt.xlabel('Iteration of Edge Removal')
    # plt.ylabel('Mean Strain')
    # plt.title('Mean Strain vs. Iteration of Edge Removal')
    # plt.savefig(f"R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Test_Lauren\\Manual_Segmentations\\Strain\\{subj}\\{dist}mi_mean_strain_iteration.png")
    # plt.close()

    if output:
        visualize_strain_map(strain_map, "strain post-removal")
        visualize_strain_map(thick_pre_map, "Pre Thickness")
        visualize_strain_map(thick_post_map, "Post Thickness")

    return strain_map, cropping_list


def parameterize_cropping(strain_ptcld, strain):
    """This function takes in a strain point cloud, and applies a series of cropping to it to see how radius
    and iterations affects cropping values"""
    n = list(range(1, 16))
    radii = [2.5, 5.0, 7.5, 10.0]
    list_out = []

    # Create matplotlib figure of strain distribution
    for radius in radii:
        for i in n:
            strain_cropped_ptcld, strain_cropped = remove_outer_boundaries(strain_ptcld, strain, radius=radius, n=i)
            mean_strain = np.mean(strain_cropped)
            list_out.append([radius, i, mean_strain, strain_cropped_ptcld, strain_cropped])
    return list_out


def visualize_strain_map(strain_map, comp_type):
    """Takes in a nx4 strain point cloud, and plots the strain surface, with label specified by comp_type"""
    coords = strain_map[:, 0:3]
    strain = strain_map[:, 3]
    strain_cloud = pv.PolyData(np.transpose([coords[:, 0], coords[:, 1], coords[:, 2]]))

    surf = strain_cloud.delaunay_2d(alpha=2)
    surf[comp_type] = strain

    if "hick" in comp_type:
        surf.plot(show_edges=False, cmap="jet")
    else:
        surf.plot(show_edges=False, cmap="seismic")
    return


def store_registered_points(registered_points, comp_type, strain_map, moving_p_ptcld, fixed_p_ptcld, moving_pc_ptcld, fixed_pc_ptcld):
    registered_points[comp_type] = {}

    registered_points[comp_type]["strain_map"] = strain_map
    registered_points[comp_type]["pre_p_ptcld"] = np.asarray(fixed_p_ptcld.points)
    registered_points[comp_type]["post_p_ptcld"] = moving_p_ptcld
    registered_points[comp_type]["pre_pc_ptcld"] = np.asarray(fixed_pc_ptcld.points)
    registered_points[comp_type]["post_pc_ptcld"] = np.asarray(moving_pc_ptcld.points)

    return registered_points


def save_registered_point_clouds(registered_points):
    with open("registered_points.pkl", 'wb') as f:
        pickle.dump(registered_points, f)


def plot_thickness_array(pc_array, scan_name, save_location, map_type, met_type):
    """Plots a pyvista map of the thickness given nx4 ndarray"""
    # Convert ptcld to pv object with strain values
    coords = np.asarray(pc_array[:, 0:3])
    thickness = pc_array[:, 3]

    max_rng = np.max(np.abs(thickness))

    strain_cloud = pv.PolyData(np.transpose([coords[:, 0], coords[:, 1], coords[:, 2]]))
    strain_cloud[met_type] = thickness
    surf = strain_cloud.delaunay_2d()

    plotter = pv.Plotter(off_screen=True)
    if "rain" in map_type:
        plotter.add_mesh(surf, scalars=met_type, cmap='seismic', rng=[-max_rng, max_rng])
        plotter.add_text(f"Mean Strain: {np.mean(thickness):.4f}", font_size=16, color="black", position='upper_right')
    else:
        plotter.add_mesh(surf, scalars=met_type, cmap='jet')
        plotter.add_text(f"Mean Thickness: {np.mean(thickness):.3f}", font_size=16, color="black")

    plotter.camera.SetPosition(np.mean(coords, axis=0) + np.array([0, 100, 0]))
    plotter.camera.SetFocalPoint(np.mean(coords, axis=0))
    plotter.camera.SetViewUp([0, 0, 1])
    plotter.show_axes()

    save_path = get_data_path("results")
    save_path = os.path.split(save_path)[:-1]
    save_path = os.path.join(*save_path, 'results', save_location)

    if not os.path.exists(os.path.join(save_path, scan_name[:11])):
        os.mkdir(os.path.join(save_path, scan_name[:11]))

    plotter.show(screenshot=os.path.join(save_path, scan_name[:11], f"{scan_name}_{map_type}.png"))
    return


if __name__ == "__main__":
    # Load in point clouds for predicted subjects
    transformations = {}
    registered_points = {}

    point_clouds = load_point_cloud()
    subj_names = list(point_clouds.keys())
    strain_vals = list()
    comp_types = ["pre1-post3mi", "pre1-rec1", "pre2-post10mi", "pre2-rec2"]

    for idx, comp_type in zip([1, 2, 4, 5], comp_types):

        if idx < 3:
            fixed_idx = 0
        else:
            fixed_idx = 3

        # Fixed ptcld for patella and patellar cartilage
        fixed_p_ptcld, fixed_p_right_ptcld = get_patella_ptclds(point_clouds, subj_names[fixed_idx])
        fixed_pc_ptcld, fixed_thickness = get_cartilage_ptcld(point_clouds, subj_names[fixed_idx])

        # Moving ptcld for patella and patellar cartilage
        moving_p_ptcld, moving_p_right_ptcld = get_patella_ptclds(point_clouds, subj_names[idx])
        moving_pc_ptcld, moving_thickness = get_cartilage_ptcld(point_clouds, subj_names[idx])

        # Perform Registration and get transformation
        moving_p_ptcld, icp_transform = move_patella(np.asarray(fixed_p_ptcld.points), np.asarray(moving_p_ptcld.points), output=False)

        # Transform other moving structures
        moving_p_right_ptcld.transform(icp_transform)
        moving_pc_ptcld.transform(icp_transform)

        # View registered point clouds
        # Resulting Patella registration
        # moving_p_ptcld.paint_uniform_color([1, 0.706, 0])
        # fixed_p_ptcld.paint_uniform_color([0, 0.651, 0.929])
        # o3d.visualization.draw_geometries([moving_p_ptcld, fixed_p_ptcld])
        #
        # Resulting Patellar cartilage surface registration
        # moving_pc_ptcld.paint_uniform_color([1, 0.706, 0])
        # fixed_pc_ptcld.paint_uniform_color([0, 0.651, 0.929])
        # o3d.visualization.draw_geometries([moving_pc_ptcld, fixed_pc_ptcld])

        # Store transformations
        transformations = store_transformations(transformations, subj_names[idx], icp_transform)

        # Create strain maps
        strain_map = produce_strain_map(moving_pc_ptcld, moving_thickness, fixed_pc_ptcld, fixed_thickness, output=True)

        # Store the strain and bone maps
        registered_points = store_registered_points(registered_points, comp_type, strain_map,
                                                    moving_p_ptcld, fixed_p_ptcld, moving_pc_ptcld, fixed_pc_ptcld)

        visualize_strain_map(strain_map, comp_type)

    # Save registered point clouds
    save_registered_point_clouds(registered_points)
