import pickle
import os
import numpy as np
import open3d as o3d
import copy


def move_patella(p_fixed, p_moving):
    # RANSAC
    # Downsample
    voxel_size_d = 1.5  # Downsample so every voxel is _mm
    p_fixed_d = p_fixed.voxel_down_sample(voxel_size=voxel_size_d)
    p_moving_d = p_moving.voxel_down_sample(voxel_size=voxel_size_d)

    # Visualize Downsampling
    # p_fixed.paint_uniform_color([1, 0.706, 0])
    # p_fixed_d.paint_uniform_color([0, 0.651, 0.929])
    # o3d.visualization.draw_geometries([p_fixed, p_fixed_d])

    # Compute normals
    search_radius_norm = voxel_size_d * 2
    p_fixed_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius_norm, max_nn=30))
    p_moving_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius_norm, max_nn=30))

    # Compute fpfh features for each point
    search_radius_feature = voxel_size_d * 5
    p_fixed_fpfh = o3d.pipelines.registration.compute_fpfh_feature(p_fixed_d, o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius_feature, max_nn=100))
    p_moving_fpfh = o3d.pipelines.registration.compute_fpfh_feature(p_moving_d, o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius_feature, max_nn=100))

    # RANSAC Execution
    distance_threshold = voxel_size_d * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(p_moving_d, p_fixed_d,
                                                                                      p_moving_fpfh, p_fixed_fpfh, True,
                                                                                      distance_threshold,
                                                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(
                                                                                          False),
                                                                                      3,
                                                                                      [
                                                                                          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                                                                                              distance_threshold)],

                                                                                      o3d.pipelines.registration.RANSACConvergenceCriteria(
                                                                                          100000, 0.9999))
    # Moving the patella
    p_moved = copy.deepcopy(p_moving)
    p_moved.transform(result.transformation)

    # Visualizing RANSAC transform
    # p_moved.paint_uniform_color([1, 0.706, 0])
    # p_fixed.paint_uniform_color([0, 0.651, 0.929])
    # o3d.visualization.draw_geometries([p_moved, p_fixed], window_name="RANSAC Result")

    # ICP Registration
    # Figure out rough estimate of voxel size
    distances = p_fixed.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # Parameters for icp
    voxel_size = 0.2
    threshold = voxel_size * 3
    initial_moving_to_fixed = result.transformation

    icp = o3d.pipelines.registration.registration_icp(p_moving, p_fixed, threshold, initial_moving_to_fixed,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      o3d.pipelines.registration.ICPConvergenceCriteria(
                                                          relative_fitness=0.000000001,
                                                          relative_rmse=0.0000000001,
                                                          max_iteration=50000)
                                                      )

    p_moved = p_moving.transform(icp.transformation)

    # Visualizing ICP transform
    # p_moved.paint_uniform_color([1, 0.706, 0])
    # p_fixed.paint_uniform_color([0, 0.651, 0.929])
    # o3d.visualization.draw_geometries([p_moved, p_fixed], window_name="ICP Result")

    return p_moved, icp.transformation


if __name__ == "__main__":
    filename = "point_clouds.pkl"
    with open(filename, "rb") as f:
        point_clouds = pickle.load(f)

    subj_names = list(point_clouds.keys())

    # Fixed and moving patella
    p_points_fixed = point_clouds[subj_names[0]]["p_coords_array"]
    p_fixed = o3d.geometry.PointCloud()
    p_fixed.points = o3d.utility.Vector3dVector(p_points_fixed)

    p_points_moving = point_clouds[subj_names[5]]["p_coords_array"]
    p_moving = o3d.geometry.PointCloud()
    p_moving.points = o3d.utility.Vector3dVector(p_points_moving)

    p_moved, icp_transform = move_patella(p_fixed, p_moving)
