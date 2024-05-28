import numpy as np
import matplotlib.pyplot as plt
import pickle
import open3d as o3d


if __name__ == "__main__":
    # Load in point clouds
    with open("registered_points.pkl", 'rb') as f:
        registered_points = pickle.load(f)

    comp_type = "pre1-post3mi"

    pre = registered_points[comp_type]["pre_p_ptcld"]
    post = registered_points[comp_type]["post_p_ptcld"]
    strain_map = registered_points[comp_type]["strain_map"]

    # Calculate mean of pre, subtract from pre and post point clouds
    mean_points = np.mean(pre, axis=0)
    pre = pre - mean_points
    post = post - mean_points
    strain_map[:, 0:3] = strain_map[:, 0:3] - mean_points

    # Perform PCA on pre, rotate pre and post. Determine if inferior-superior flipped or not.
    U, S, Vt = np.linalg.svd(pre)
    print(Vt.T)
    pre_rot = pre @ Vt.T
    post_rot = post @ Vt.T

    pre_rot_ptcld = o3d.geometry.PointCloud()
    pre_rot_ptcld.points = o3d.utility.Vector3dVector(pre_rot)

    post_rot_ptcld = o3d.geometry.PointCloud()
    post_rot_ptcld.points = o3d.utility.Vector3dVector(post_rot)

    # Visualize
    pre_rot_ptcld.paint_uniform_color([1, 0, 0])
    post_rot_ptcld.paint_uniform_color([0, 0, 1])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=40, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pre_rot_ptcld, post_rot_ptcld, mesh_frame])

    # Identify centroid of pre patella surface on articulating patella surface (extended in z dimension)
    # Move further along the z dimension to the strain map surface
    # Move along x and y dimensions on patella to place rest of nodes
