import numpy as np
import pickle
from strain_analysis import load_point_cloud



if __name__ == "__main__":
    point_clouds = load_point_cloud()

    np.savetxt('pre_P.txt', point_clouds["AS_006"]["p_coords_array"], delimiter='\t', fmt='%.6f')
    np.savetxt('post_P.txt', point_clouds["AS_007"]["p_coords_array"], delimiter='\t', fmt='%.6f')

    np.savetxt('pre_PC.txt', point_clouds["AS_006"]["pc_coords_array"], delimiter='\t', fmt='%.6f')
    np.savetxt('post_PC.txt', point_clouds["AS_007"]["pc_coords_array"], delimiter='\t', fmt='%.6f')