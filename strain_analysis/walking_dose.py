# This script analyzes the data from Paranjape et al. 2020
import numpy as np
import os
import pickle
import tensorflow as tf
import itertools
import open3d as o3d
import matplotlib.pyplot as plt

from dice_loss_function import dice_loss
from get_data_path import get_data_path
from evaluate import load_model
from point_clouds import get_coordinate_arrays, calculate_thickness
from registration import move_patella, move_point_cloud
from strain_analysis import produce_strain_map
from PIL import Image

def get_paranjape_dataset(image_dir, batch_size):
    """Returns a tf.data.Dataset object containing the filenames of the Paranjape dataset"""
    dataset = tf.data.Dataset.list_files(image_dir, shuffle=False)
    dataset = dataset.map(load_data)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def load_data(image_path):
    """Load in MRI slice and return the slice tensor and the filename"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image)
    image = tf.cast(image, tf.float64) / 255.0
    filename = tf.strings.split(image_path, os.path.sep)[-1]

    return filename, image


def get_model_filename(model_name):
    """Assembles model absolute filename from model name"""
    model_filename = os.path.join("..", "models", model_name)
    return model_filename


def process_predicted_label(pred_label):
    """Takes in (batch_size, 256, 256, 2) ndarray, and returns two, (256, 256, 12) binary ndarrays of P and PC preds"""

    thresholded_label = (pred_label >= 0.5)
    binary_data = thresholded_label.astype(np.uint8)

    pat = np.squeeze(binary_data[:, :, :, 0])
    pat_cart = np.squeeze(binary_data[:, :, :, 1])

    # Transpose to (256, 256, batch_size)
    pat = np.transpose(pat, (1, 2, 0))
    pat_cart = np.transpose(pat_cart, (1, 2, 0))

    return pat, pat_cart


def parse_scan_name(filename):
    """Takes in tf.tensor of b-strings and returns the scan name (str). Also, checks that the first image is here"""
    filename_str_start = filename.numpy()[0].decode()

    # Remove extension, get slice number, assert it is 1, and return the scan name
    no_ext = filename_str_start.split(".")[0]
    num = int(no_ext.split("-")[1])
    assert num == 26, (f"Images are out of order! First image in this batch is not first in scan! "
                      f"First batch image: {filename_str_start}")

    scan_name = no_ext.split("-")[0]

    print(f"Starting scan {scan_name}")

    return scan_name


def save_volumes(pat_vol, pat_cart_vol, scan_name):
    save_path = "R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Test_Lauren\\Manual_Segmentations\\Volumes"
    # Save scans
    np.savez(os.path.join(save_path, scan_name), pat_vol=pat_vol, pat_cart_vol=pat_cart_vol)
    return


def load_volumes(scan, volume_path):
    """Loads in .npz volume predictions and returns p_vol and pc_vol"""
    loaded_data = np.load(os.path.join(volume_path, scan))
    pat_vol = loaded_data["pat_vol"]
    pat_cart_vol = loaded_data["pat_cart_vol"]

    return pat_vol, pat_cart_vol


def save_coordinate_arrays(p_array, pc_array, scan):
    """Writes ndarrays to point_cloud path"""
    save_path = "R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Test_Lauren\\Manual_Segmentations\\To_Geomagic"

    # Remove .npz extension on scan, if there
    if scan[-4:] == ".npz":
        scan = scan[:-4]

    np.savetxt(f"{get_data_path('Paranjape_ToGeomagic')}/P/{scan}_P.txt", p_array, delimiter='\t', fmt='%.6f')
    np.savetxt(f"{get_data_path('Paranjape_ToGeomagic')}/PC/{scan}_PC.txt", pc_array, delimiter='\t', fmt='%.6f')

    print(f"Saved {scan} point clouds")
    return


def load_coordinate_arrays(scan):
    """Loads in .pcd point clouds from Geomagic
    Geomagic exports in meters, so we'll convert all distances to millimeters"""

    # Remove .npz extension on scan, if there
    if scan[-4:] == ".npz":
        scan = scan[:-4]

    filename_P = f"{get_data_path('Paranjape_FromGeomagic')}/{scan}_P.pcd"
    filename_PC = f"{get_data_path('Paranjape_FromGeomagic')}/{scan}_PC.pcd"

    ptcld_P = o3d.io.read_point_cloud(filename_P)
    ptcld_PC = o3d.io.read_point_cloud(filename_PC)

    p_array = np.asarray(ptcld_P.points) * 1000.0
    pc_array = np.asarray(ptcld_PC.points) * 1000.0

    return p_array, pc_array


def scan_properties(scan2, scan1, info, strain):
    """Takes in two names of scans and asserts that they're pre and post, while extracting info from the scan name"""
    assert "pre" in scan2.lower(), f"Scan {scan2} is not a pre scan"
    assert "post" in scan1.lower(), f"Scan {scan1} is not a post scan"

    # Extract information from this pair of scans
    subject_id = scan2[0:2]
    froude = scan2[3:6]
    duration = scan2[7:9]

    info["Subject ID"].append(subject_id)
    info["Froude"].append(froude)
    info["Duration"].append(duration)
    info["Mean Strain"].append(strain)
    return info


def create_point_clouds(pre_pc_array, post_pc_array):
    """Takes in (n, 3) cartilage surface point clouds and creates o3d point cloud objects"""
    pre_pc_ptcld = o3d.geometry.PointCloud()
    pre_pc_ptcld.points = o3d.utility.Vector3dVector(pre_pc_array)

    post_pc_ptcld = o3d.geometry.PointCloud()
    post_pc_ptcld.points = o3d.utility.Vector3dVector(post_pc_array)
    return pre_pc_ptcld, post_pc_ptcld


def output_plots(info):
    froude_categories = ["010", "025", "040"]
    duration_categories = ["10", "20", "30", "40", "60"]

    foude_data = {category: [] for category in froude_categories}
    duration_data = {category: [] for category in duration_categories}

    for froude, duration, strain in zip(info["Froude"], info["Duration"], info["Mean Strain"]):
        if duration == "30":
            foude_data[froude].append(strain)
        if froude == "025":
            duration_data[duration].append(strain)

    # Convert the data to a list of lists for boxplot
    froude_list = [foude_data[category] for category in froude_categories]
    duration_list = [duration_data[category] for category in duration_categories]

    # Create froude vs strain plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(froude_list, labels=froude_categories)

    # Customize the plot
    plt.xlabel("Froude")
    plt.ylabel("Mean Strain")
    plt.title("Mean Strain vs Froude (Duration = 30 min)")

    # Show the plot
    plt.show()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(duration_list, labels=duration_categories)

    # Customize the plot
    plt.xlabel("Duration (min)")
    plt.ylabel("Mean Strain")
    plt.title("Mean Strain vs Duration (Froude = 0.25)")

    # Show the plot
    plt.show()


def read_in_labeled_data(subj_dir, subj):
    p_path = os.path.join(subj_dir, subj, "P")
    pc_path = os.path.join(subj_dir, subj, "PC")

    p_vol = np.zeros((512, 512, 70), dtype=np.uint8)
    pc_vol = np.zeros((512, 512, 70), dtype=np.uint8)
    j = 0

    files = os.listdir(p_path)
    for idx, file in enumerate(files):
        if idx < 95 and idx >= 25:
            p_file = os.path.join(p_path, file)
            pc_file = os.path.join(pc_path, file)

            p_img = np.asarray(Image.open(p_file)) / 255.0
            pc_img = np.asarray(Image.open(pc_file)) / 255.0

            p_vol[:, :, j] = p_img.astype(np.uint8)
            pc_vol[:, :, j] = pc_img.astype(np.uint8)
            j += 1

    return p_vol, pc_vol


if __name__ == "__main__":
    # Options:
    predict_volumes_option = True
    create_point_clouds_option = True
    # Geomagic here
    register_point_clouds_option = False
    visualize_registration_option = False
    visualize_strain_map_option = False

    # Post, pre pairs
    scans = ["AS_001", "AS_002", "AS_003", "AS_004", "AS_007", "AS_008", "AS_010", "AS_011", "AS_013", "AS_012", "AS_016", "AS_015",
             "AS_019", "AS_018", "AS_022", "AS_021", "AS_026", "AS_025", "AS_029", "AS_028", "AS_034", "AS_030", "AS_031", "AS_035",
             "AS_037", "AS_036", "AS_041", "AS_038", "AS_047", "AS_043"]
    dist = ["3", "10", "10", "3", "10", "3", "3", "10", "3", "10", "3", "10", "10", "3", "10"]

    # Predict patella and patellar cartilage volumes
    if predict_volumes_option:

        subj_dir = "R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Organized_Data"
        subjs = os.listdir(subj_dir)
        subjs = subjs[35:]
        for idx, subj in enumerate(subjs):
            if idx < 48:
                p_vol, pc_vol = read_in_labeled_data(subj_dir, subj)
                save_volumes(p_vol, pc_vol, subj)
                print(f"saved {subj} volumes")

    # Create point clouds of the patella, patellar cartilage, and articulating surface of patella
    if create_point_clouds_option:
        # Get volume data path (what we're reading in)
        volume_path = "R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Test_Lauren\\Manual_Segmentations\\Volumes"
        scans = os.listdir(volume_path)

        # Create point cloud data path (what we're saving to)
        point_cloud_path = "R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Test_Lauren\\Manual_Segmentations\\To_Geomagic"

        # Iterate through each volume, calculate coordinate arrays, and save
        for scan in scans:
            pat_vol, pat_cart_vol = load_volumes(scan, volume_path)
            p_array, pc_array = get_coordinate_arrays(pat_vol, pat_cart_vol)
            save_coordinate_arrays(p_array, pc_array, scan)

    # Load pre and post, register, calculate strain map, save registered point clouds and strain map
    if register_point_clouds_option:
        # Get point cloud data path (what we're reading in)
        point_cloud_path = "R:\\DefratePrivate\\Bercaw\\Patella_Autoseg\\Test_Lauren\\Manual_Segmentations\\From_Geomagic"
        scans = os.listdir(point_cloud_path)

        # Get strain data path (what we're saving to)
        strain_path = os.path.join(get_data_path("Paranjape_PCs"))
        if not os.path.exists(strain_path):
            os.mkdir(strain_path)

        # Initialize info
        info = {}
        info["Subject ID"] = []
        info["Froude"] = []
        info["Duration"] = []
        info["Mean Strain"] = []

        # Iterate through each scan and take in a pair of scans
        for idx in range(len(scans)):
            if idx % 2 != 0:
                print(f"Current Scans: {scans[idx]} and {scans[idx-1]}")
                pre_p_array, pre_pc_array = load_coordinate_arrays(scans[idx])
                post_p_array, post_pc_array = load_coordinate_arrays(scans[idx-1])

                # Calculate thickness
                pre_pc_array = calculate_thickness(pre_p_array, pre_pc_array)
                post_pc_array = calculate_thickness(post_p_array, post_pc_array)

                # Extract thicknesses
                pre_thickness = pre_pc_array[:, 3]
                post_thickness = post_pc_array[:, 3]

                # Remove last column on pc_arrays
                pre_pc_array = pre_pc_array[:, :-1]
                post_pc_array = post_pc_array[:, :-1]

                # Register the patella
                post_p_array, transform = move_patella(pre_p_array, post_p_array, output=visualize_registration_option)

                # Move other structures
                post_pc_array = move_point_cloud(post_pc_array, transform)

                # Create o3d point clouds
                pre_pc_ptcld, post_pc_ptcld = create_point_clouds(pre_pc_array, post_pc_array)

                # Calculate strain map
                strain_map = produce_strain_map(post_pc_ptcld, post_thickness, pre_pc_ptcld, pre_thickness, output=visualize_strain_map_option)

                # Calculate the strain at the middle-most point of the strain map
                # strains = list(strain_map[:, 3])
                # plt.hist(strains)
                # plt.xlabel("Strain")
                # plt.show()

                mean_strain = np.mean(strain_map[:, 3])
                print(f"Mean strain for {scans[idx]}: {mean_strain}")

                info = scan_properties(scans[idx], scans[idx - 1], info, mean_strain)

        output_plots(info)
