import numpy as np
import matplotlib.pyplot as plt
from strain_analysis import get_cartilage_ptcld, get_patella_ptclds, load_point_cloud

if __name__ == "__main__":
    # Calculate mean of pre, subtract from pre and post point clouds
    # Perform PCA on pre, rotate pre and post
    # Identify centroid of pre patella surface on articulating patella surface (extended in z dimension)
    # Move further along the z dimension to the strain map surface
    # Move along x and y dimensions on patella to place rest of nodes
