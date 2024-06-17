import pickle
import os
import numpy as np
import open3d as o3d
import copy
import imageio
import moviepy.editor as mpy


def draw_movie(p_moved, p_fixed):
    # Create 3 point clouds
    points_a = np.asarray(p_moved.points)
    points_b = np.asarray(p_fixed.points)

    # Find common points
    common_points = np.array([point for point in points_a if np.any(np.all(point == points_b, axis=1))])
    unique_a = np.array([point for point in points_a if not np.any(np.all(point == points_b, axis=1))])
    unique_b = np.array([point for point in points_b if not np.any(np.all(point == points_a, axis=1))])

    # Create new point clouds for unique and common points
    pcd_common = o3d.geometry.PointCloud()
    pcd_common.points = o3d.utility.Vector3dVector(common_points)
    pcd_common.paint_uniform_color([0, 0.5, 0])  # Dark green

    pcd_unique_a = o3d.geometry.PointCloud()
    pcd_unique_a.points = o3d.utility.Vector3dVector(unique_a)
    pcd_unique_a.paint_uniform_color([1, 0, 0])  # Red

    pcd_unique_b = o3d.geometry.PointCloud()
    pcd_unique_b.points = o3d.utility.Vector3dVector(unique_b)
    pcd_unique_b.paint_uniform_color([0, 0, 1])  # Blue

    # Visualize all point clouds together
    # o3d.visualization.draw_geometries([pcd_common, pcd_unique_a, pcd_unique_b])

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Zero mean the point clouds
    centroid = np.mean(np.asarray(p_fixed.points), axis=0)

    pcd_common.translate(-centroid)
    pcd_unique_a.translate(-centroid)
    pcd_unique_b.translate(-centroid)

    vis.add_geometry(pcd_common)
    vis.add_geometry(pcd_unique_a)
    vis.add_geometry(pcd_unique_b)

    # Get the render options and set point size
    render_option = vis.get_render_option()
    render_option.point_size = 7.0

    # Directory to save frames
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)

    num_frames = 1200  # Number of frames for the rotation
    angle_step1 = 1 * np.pi / num_frames  # Full rotation in radians
    angle_step2 = 2 * np.pi / num_frames  # Full rotation in radians

    # Rotate the view around the z-axis
    R1 = np.array([[1, 0, 0, 0],
              [0, np.cos(angle_step1), -np.sin(angle_step1), 0],
              [0, np.sin(angle_step1), np.cos(angle_step1), 0],
              [0, 0, 0, 1]])
    R2 = np.array([[np.cos(angle_step2), 0, np.sin(angle_step2), 0],
                     [0, 1, 0, 0],
                     [-np.sin(angle_step2), 0, np.cos(angle_step2), 0],
                     [0, 0, 0, 1]])
    R = np.matmul(R1, R2)

    for i in range(num_frames):

        pcd_common.transform(R)
        pcd_unique_a.transform(R)
        pcd_unique_b.transform(R)

        vis.update_geometry(pcd_common)
        vis.update_geometry(pcd_unique_a)
        vis.update_geometry(pcd_unique_b)

        vis.poll_events()
        vis.update_renderer()

        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        vis.capture_screen_image(frame_path)

    vis.destroy_window()

    # Compile the frames into a movie using imageio and moviepy
    # Load all the frames
    frame_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])
    frames = [imageio.imread(frame_file) for frame_file in frame_files]

    # Create a video
    clip = mpy.ImageSequenceClip(frames, fps=50)  # Adjust fps as needed
    clip.write_videofile("rotation_movie.mp4", codec="libx264")

    print("Movie saved as rotation_movie.mp4")
    return


def move_patella(p_points_fixed, p_points_moving, output):

    # Create open3d point cloud objects
    p_fixed = o3d.geometry.PointCloud()
    p_fixed.points = o3d.utility.Vector3dVector(p_points_fixed)

    p_moving = o3d.geometry.PointCloud()
    p_moving.points = o3d.utility.Vector3dVector(p_points_moving)

    # Before registration
    # draw_movie(p_moving, p_fixed)

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
    threshold = voxel_size * 8
    initial_moving_to_fixed = result.transformation

    icp = o3d.pipelines.registration.registration_icp(p_moving, p_fixed, threshold, initial_moving_to_fixed,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      o3d.pipelines.registration.ICPConvergenceCriteria(
                                                          relative_fitness=1e-10,
                                                          relative_rmse=1e-10,
                                                          max_iteration=50000)
                                                      )

    p_moved = p_moving.transform(icp.transformation)

    if output:
        # Visualizing ICP transform
        p_moved.paint_uniform_color([1, 0, 0])
        p_fixed.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([p_moved, p_fixed], window_name="ICP Result")

    p_moved = np.asarray(p_moved.points)

    return p_moved, icp.transformation


def move_point_cloud(post_array, transform):
    """Takes in a (n, 3) ndarray of a point cloud, and a (4, 4) transform and transforms the point cloud"""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(post_array)
    point_cloud.transform(transform)
    return np.asarray(point_cloud.points)


if __name__ == "__main__":
    filename = "point_clouds.pkl"
    with open(filename, "rb") as f:
        point_clouds = pickle.load(f)

    subj_names = list(point_clouds.keys())

    # Fixed and moving patella
    p_points_fixed = point_clouds[subj_names[3]]["p_coords_array"]

    p_points_moving = point_clouds[subj_names[4]]["p_coords_array"]

    p_moved, icp_transform = move_patella(p_points_fixed, p_points_moving, output=True)
