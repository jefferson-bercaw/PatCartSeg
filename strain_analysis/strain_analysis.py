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


def remove_outer_boundaries(pc_ptcld, thickness, radius):

    # Estimate normals for the entire point cloud
    pc_ptcld.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    # Compute point density
    kdtree = o3d.geometry.KDTreeFlann(pc_ptcld)
    densities = np.zeros(len(pc_ptcld.points))

    for i, point in enumerate(pc_ptcld.points):
        [_, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        densities[i] = len(idx)

    # plt.hist(densities, bins=15)
    # plt.show()
    # Threshold for edge removal
    density_threshold = round(np.max(densities) / 2)

    # Filter points based on density
    filtered_points = []
    filtered_thickness = []
    for i, density in enumerate(densities):
        if density >= density_threshold:
            filtered_points.append(pc_ptcld.points[i])
            filtered_thickness.append(thickness[i])

    # Create a new point cloud from the filtered points
    filtered_pc_ptcld = o3d.geometry.PointCloud()
    filtered_pc_ptcld.points = o3d.utility.Vector3dVector(np.array(filtered_points))

    return filtered_pc_ptcld, np.array(filtered_thickness)


def produce_strain_map(pc_ptcld, thickness, fixed_pc_ptcld, fixed_thickness, output):
    # Remove outer boundaries
    pc_ptcld_full = copy.deepcopy(pc_ptcld)
    fixed_pc_ptcld_full = copy.deepcopy(fixed_pc_ptcld)

    pc_ptcld, thickness = remove_outer_boundaries(pc_ptcld, thickness, radius=1.0)
    fixed_pc_ptcld, fixed_thickness = remove_outer_boundaries(fixed_pc_ptcld, fixed_thickness, radius=1.0)

    # Visualize removal
    if output:
        pc_ptcld.paint_uniform_color([1, 0, 0])
        pc_ptcld_full.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([pc_ptcld, pc_ptcld_full])
        #
        fixed_pc_ptcld.paint_uniform_color([1, 0, 0])
        fixed_pc_ptcld_full.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([fixed_pc_ptcld, fixed_pc_ptcld_full])

        # Visualize two registered thickness maps
        pc_ptcld.paint_uniform_color([1, 0, 0])
        fixed_pc_ptcld.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([pc_ptcld, fixed_pc_ptcld])

    # Visualize thickness maps
    thick_pre_map = np.concatenate((np.asarray(pc_ptcld.points), thickness[:, np.newaxis]), axis=1)
    thick_post_map = np.concatenate((np.asarray(fixed_pc_ptcld.points), fixed_thickness[:, np.newaxis]), axis=1)

    # Average thickness values over a certain area
    moving_pc, thickness = average_thickness_values(pc_ptcld, thickness)
    fixed_pc, fixed_thickness = average_thickness_values(fixed_pc_ptcld, fixed_thickness)

    # # Get points arrays
    # moving_pc = np.asarray(pc_ptcld.points)
    # fixed_pc = np.asarray(fixed_pc_ptcld.points)

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

        # Threshold distance. If the distance between these two coordinates isn't too large, add to strain map
        dist_thresh = 0.75  # distance [mm] that signifies a "good" comparison
        if np.linalg.norm(moving_coord - fixed_coord) < dist_thresh:
            pre_thick = fixed_thickness[closest_indices[i]]
            post_thick = thickness[i]

            avg_coord.append((fixed_coord + moving_coord) / 2)  # average coordinate location

            strain_here = (post_thick - pre_thick) / pre_thick
            strain.append(strain_here)

    # Convert coords and strain lists to numpy arrays
    avg_coord = np.array(avg_coord)
    strain = np.array(strain)

    # Remove outer boundaries of strain map
    strain_ptcld = o3d.geometry.PointCloud()
    strain_ptcld.points = o3d.utility.Vector3dVector(avg_coord)

    # Copy strain map for visual comparison
    strain_map_pre_removal = np.concatenate((np.asarray(strain_ptcld.points), strain[:, np.newaxis]), axis=1)

    # Remove outer boundaries
    strain_ptcld, strain = remove_outer_boundaries(strain_ptcld, strain, radius=3.0)

    strain_map = np.concatenate((np.asarray(strain_ptcld.points), strain[:, np.newaxis]), axis=1)
    #
    # plt.hist(strain)
    # plt.show()

    if output:
        visualize_strain_map(thick_pre_map, "Pre Thickness")
        visualize_strain_map(thick_post_map, "Post Thickness")
        visualize_strain_map(strain_map_pre_removal, "strain pre-removal")
        visualize_strain_map(strain_map, "strain post-removal")

    return strain_map


def visualize_strain_map(strain_map, comp_type):
    """Takes in a nx4 strain point cloud, and plots the strain surface, with label specified by comp_type"""
    coords = strain_map[:, 0:3]
    strain = strain_map[:, 3]
    strain_cloud = pv.PolyData(np.transpose([coords[:, 0], coords[:, 1], coords[:, 2]]))

    surf = strain_cloud.delaunay_2d(alpha=2)
    surf[comp_type] = strain

    if "hick" in comp_type:
        surf.plot(show_edges=False, cmap="plasma", rng=[0, 9])
    else:
        surf.plot(show_edges=False, cmap="plasma", rng=[-0.3, 0.3])
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
