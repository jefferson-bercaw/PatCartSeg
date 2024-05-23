import pickle
import os
import numpy as np
import open3d as o3d


if __name__ == "__main__":
    filename = "point_clouds.pkl"
    with open(os.path.join("history", filename), "rb") as f:
        pickle.load(f)
