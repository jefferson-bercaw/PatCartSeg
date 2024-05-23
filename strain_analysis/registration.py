import pickle
import os
import numpy as np
import open3d as o3d


if __name__ == "__main__":
    filename = "point_clouds.pkl"
    with open(filename, "rb") as f:
        point_clouds = pickle.load(f)

    subj_names = list(point_clouds.keys())

    # Fixed and moving patella
    p_points = point_clouds[subj_names[0]]["p_coords_array"]
    p_ptcld_fixed = o3d.geometry.PointCloud()
    p_ptcld_fixed.points = o3d.utility.Vector3dVector(p_points)

    p_points_moving = point_clouds[subj_names[1]]["p_coords_array"]
    p_ptcld_moving = o3d.geometry.PointCloud()
    p_ptcld_moving.points = o3d.utility.Vector3dVector(p_points)

    # Visualize initial positioning
    p_ptcld_fixed.paint_uniform_color([1, 0.706, 0])
    p_ptcld_moving.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([p_ptcld_fixed, p_ptcld_moving])

    #
    print("Done")

