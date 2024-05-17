import open3d as o3d
import numpy as np
import os



##Building import and export file paths
###Input bone and perameters as listed in file names (ie PreTibia)
tissue = 'PreFemur'
###Input file path directory
directory='R:\DefratePrivate\Otap\StrainAnalysisTestingFiles\R2K064PyRegTesting\Visit 1\Coordinates'
#
#Adding filename suffix
filenametxt = tissue + '.txt'
filenameply = tissue + '.ply'
#Input txt file path
filepathtxt = os.path.join(directory,filenametxt)
print(filepathtxt)
#Output ply file path
filepathply = os.path.join(directory, filenameply)
print(filepathply)



#Function for loading point cloud data from .txt
def load_txt(filepath):
    points = []
    with open(filepath, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x,y,z])
    return np.asarray(points)


##Processing Steps
#Converting Txt into NumPy array
pc_data = load_txt(filepathtxt)
#Creating Open3D PointCloud from NumPy array
pcloud = o3d.geometry.PointCloud()
pcloud.points = o3d.utility.Vector3dVector(pc_data)



##Plotting (optional) and Exporting steps
#Visualize Open3D PointCloud (optional)
o3d.visualization.draw_geometries(([pcloud]))
#Exporting as PLY file
o3d.io.write_point_cloud(filepathply, pcloud)