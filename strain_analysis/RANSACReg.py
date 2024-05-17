import open3d as o3d
import numpy as np
import copy

#Downsampling and estimating normals, required for RANSAC strain_analysis
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

#Function for running RANSAC(initial) strain_analysis
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC strain_analysis on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

#Point-to-Plan ICP Registration Function
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.5
    print(":: Point-to-plane ICP strain_analysis is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


if __name__ == "__main__":
    # Load the pre (target) point cloud
    target_pcd = o3d.io.read_point_cloud("R:\DefratePrivate\Otap\StrainAnalysisTestingFiles\R2K064PyRegTesting\Visit 1\Coordinates\PreTibia.ply")
    #
    # # Load the post (moving) point cloud
    source_pcd = o3d.io.read_point_cloud("R:\DefratePrivate\Otap\StrainAnalysisTestingFiles\R2K064PyRegTesting\Visit 1\Coordinates\PostTibia.ply")
    # print(type(source_pcd))

    # o3d.visualization.draw_geometries([source_pcd])
    # Voxel Size for down sampling (Adjust if the Ransac strain_analysis is poor)
    voxel_size = 0.5

    #Running Preprocessing function
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    # o3d.visualization.draw_geometries([source_down])

    #Running RANSAC function
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    print(result_ransac.transformation)

    source_temp = copy.deepcopy(source_pcd)
    target_temp = copy.deepcopy(target_pcd)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(result_ransac.transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

    #Running
    result_icp = refine_registration(source_pcd, target_pcd, source_fpfh, target_fpfh,
                                     voxel_size)
    print(result_icp)

    source_temp = copy.deepcopy(source_pcd)
    target_temp = copy.deepcopy(target_pcd)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(result_icp.transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

