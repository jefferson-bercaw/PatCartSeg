import numpy as np
import open3d as o3d
import pickle
import os


def load_point_cloud():
    with open(os.path.join(os.getcwd(), "strain_analysis", "point_clouds.pkl"), 'rb') as f:
        point_clouds = pickle.load(f)
    return point_clouds


if __name__ == "__main__":
    # Load in point clouds for predicted subjects
    point_clouds = load_point_cloud()

    print("Done")