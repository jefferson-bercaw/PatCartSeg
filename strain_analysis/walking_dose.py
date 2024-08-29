# This script analyzes the data from Paranjape et al. 2020
import numpy as np
import os
import pickle
import tensorflow as tf
import itertools
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import csv
import pyvista as pv
import pandas as pd

from dice_loss_function import dice_loss
from get_data_path import get_data_path
from evaluate import load_model
from point_clouds import get_coordinate_arrays, calculate_thickness
from registration import move_patella, move_point_cloud
from strain_analysis import produce_strain_map, plot_thickness_array
from randomization import randomize, derandomize


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
    assert num == 21, (f"Images are out of order! First image in this batch is not first in scan! "
                      f"First batch image: {filename_str_start}")

    scan_name = no_ext.split("-")[0]

    print(f"Starting scan {scan_name}")

    return scan_name


def reshape_mri(mri):
    """Takes in (batch_size, 256, 256) MRI tensor and reshapes it to (256, 256, batch_size)"""
    mri = mri.numpy()
    mri = mri.reshape(10, 256, 256)
    mri = mri.transpose(1, 2, 0)
    return mri


def save_bmps(mri_vol, pat_vol, pat_cart_vol, model_name, scan_name):
    start_path = get_data_path("Crook_Predictions")
    mri_path = os.path.join(start_path, model_name[0:-3], "original", scan_name, "mri")
    pat_path = os.path.join(start_path, model_name[0:-3], "original", scan_name, "ML_pat")
    pat_cart_path = os.path.join(start_path, model_name[0:-3], "original", scan_name, "ML_patcart")

    # If paths don't exist, create them
    if not os.path.exists(mri_path):
        os.makedirs(mri_path)
    if not os.path.exists(pat_path):
        os.makedirs(pat_path)
    if not os.path.exists(pat_cart_path):
        os.makedirs(pat_cart_path)

    # Assert shape is correct
    assert mri_vol.shape == (256, 256, 80), (f"MRI volume being saved is not (256, 256, 80)! "
                                                f"Shape {mri_vol.shape}")
    assert pat_vol.shape == (256, 256, 80), (f"Patella volume being saved is not (256, 256, 80)! "
                                                f"Shape {pat_vol.shape}")
    assert pat_cart_vol.shape == (256, 256, 80), (f"Patellar cartilage volume being saved is not (256, 256, 80)! "
                                                f"Shape {pat_cart_vol.shape}")

    # Save scans
    for i in range(mri_vol.shape[2]):
        mri_filename = os.path.join(mri_path, f"{scan_name}-{i+21:04}.bmp")
        pat_filename = os.path.join(pat_path, f"{scan_name}-{i+21:04}.bmp")
        pat_cart_filename = os.path.join(pat_cart_path, f"{scan_name}-{i+21:04}.bmp")

        mri = 255 * mri_vol[:, :, i]
        pat = pat_vol[:, :, i]
        pat_cart = pat_cart_vol[:, :, i]

        mri = mri.astype(np.uint8)
        pat = (pat * 255).astype(np.uint8)
        pat_cart = (pat_cart * 255).astype(np.uint8)

        mri_img = Image.fromarray(mri)
        pat_img = Image.fromarray(pat)
        pat_cart_img = Image.fromarray(pat_cart)

        mri_img.save(mri_filename)
        pat_img.save(pat_filename)
        pat_cart_img.save(pat_cart_filename)


def save_volumes(pat_vol, pat_cart_vol, model_name, scan_name):
    start_path = get_data_path("Crook_Volumes")
    volume_path = os.path.join(start_path, model_name[0:-3])

    # If paths don't exist, create them
    if not os.path.exists(start_path):
        os.mkdir(start_path)
    if not os.path.exists(volume_path):
        os.mkdir(volume_path)

    # Assert shape is correct
    assert pat_vol.shape == (256, 256, 80), (f"Patella volume being saved is not (256, 256, 80)! "
                                             f"Shape {pat_vol.shape}")
    assert pat_cart_vol.shape == (256, 256, 80), (f"Patellar cartilage volume being saved is not (256, 256, 80)! "
                                                f"Shape {pat_cart_vol.shape}")

    # Save scans
    np.savez(os.path.join(volume_path, scan_name), pat_vol=pat_vol, pat_cart_vol=pat_cart_vol)
    return


def load_volumes(scan, volume_path):
    """Loads in .npz volume predictions and returns p_vol and pc_vol"""
    loaded_data = np.load(os.path.join(volume_path, scan))
    pat_vol = loaded_data["pat_vol"]
    pat_cart_vol = loaded_data["pat_cart_vol"]

    return pat_vol, pat_cart_vol


def save_coordinate_arrays(p_array, pc_array, scan):
    """Writes ndarrays to point_cloud path"""

    # Remove .npz extension on scan, if there
    if scan[-4:] == ".npz":
        scan = scan[:-4]

    np.savetxt(f"{get_data_path('Crook_ToGeomagic')}\\P\\{scan}_P.txt", p_array, delimiter='\t', fmt='%.6f')
    np.savetxt(f"{get_data_path('Crook_ToGeomagic')}\\PC\\{scan}_PC.txt", pc_array, delimiter='\t', fmt='%.6f')

    print(f"Saved {scan} point clouds")
    return


def load_coordinate_arrays(scan):
    """Loads in .pcd point clouds from Geomagic
    Geomagic exports in meters, so we'll convert all distances to millimeters"""

    # Remove .npz extension on scan, if there
    if scan[-4:] == ".npz":
        scan = scan[:-4]

    filename_P = f"{get_data_path('Crook_FromGeomagic')}/{scan}_P.pcd"
    filename_PC = f"{get_data_path('Crook_FromGeomagic')}/{scan}_PC.pcd"

    ptcld_P = o3d.io.read_point_cloud(filename_P)
    ptcld_PC = o3d.io.read_point_cloud(filename_PC)

    p_array = np.asarray(ptcld_P.points) * 1000.0
    pc_array = np.asarray(ptcld_PC.points) * 1000.0

    # Raw point clouds
    # filename_P = f"{get_data_path('Paranjape_ToGeomagicSpeed')}/P/{scan}_P.txt"
    # filename_PC = f"{get_data_path('Paranjape_ToGeomagicSpeed')}/PC/{scan}_PC.txt"
    #
    # p_array = np.loadtxt(filename_P, delimiter='\t')
    # pc_array = np.loadtxt(filename_PC, delimiter='\t')

    return p_array, pc_array


def scan_properties(scan2, scan1, info, strain):
    """Takes in two names of scans and asserts that they're pre and post, while extracting info from the scan name"""
    assert "pre" in scan2.lower(), f"Scan {scan2} is not a pre scan"
    assert "post" in scan1.lower(), f"Scan {scan1} is not a post scan"

    # Extract information from this pair of scans
    subject_id = scan2[0:8]

    info["Subject ID"].append(subject_id)
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
    """Output boxplots of strain vs deficiency group (ACL Deficiency vs. Healthy Contralateral)"""
    groups = info["Group"]
    strain = info["Mean Strain"]
    dthick = info["Change_in_Thickness"]

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({"Group": groups, "Mean Strain": strain, "Change in Thickness": dthick})

    # Plot a box plot of change in thickness in each of the two groups
    plt.figure(figsize=(10, 6))
    df.boxplot(column="Change in Thickness", by="Group", grid=False)

    # Customize the plot
    plt.xlabel("Group")
    plt.ylabel("Change in Thickness")
    plt.title("Change in Thickness for each group")
    plt.suptitle('')  # Suppress the automatic title from pandas boxplot

    # Add horizontal line along y = 0 across both bars
    plt.axhline(y=0, color='black', linestyle='--')

    # Show the plot
    plt.show()

    # Plot a box plot of strain in each of the two groups
    plt.figure(figsize=(10, 6))
    df.boxplot(column="Mean Strain", by="Group", grid=False)

    # Customize the plot
    plt.xlabel("Group")
    plt.ylabel("Mean Strain")
    plt.title("Mean strain for each group")
    plt.suptitle('')  # Suppress the automatic title from pandas boxplot

    # Add horizontal line along y = 0 across both bars
    plt.axhline(y=0, color='black', linestyle='--')

    # Show the plot
    plt.show()

    return


def scan_criteria(scan, excluded_list):
    """Returns True if the scan meets the criteria for analysis"""
    for excluded in excluded_list:
        if excluded in scan:
            return False
    return True


def save_cropping_list(scan_name, cropping_list, save_location):
    """Saves a list of cropping values to an excel row"""
    save_path = get_data_path("results")
    save_path = os.path.split(save_path)[:-1]
    save_path = os.path.join(*save_path, 'results', save_location)
    save_file = os.path.join(save_path, "cropping_list.csv")

    header1 = ["-"] + ["Radius (mm)"] + [item[0] for item in cropping_list]
    header2 = ["Duration"] + ["Iterations (n)"] + [item[1] for item in cropping_list]
    row = [scan_name[7:9]] + [scan_name] + [item[2] for item in cropping_list]

    rads = [item[0] for item in cropping_list]
    iterations = [item[1] for item in cropping_list]
    mean_strains = [item[2] for item in cropping_list]
    ptclds = [item[3] for item in cropping_list]
    strains = [item[4] for item in cropping_list]

    # Iterate through each scan and save .pngs of pyvista plot under the right folder
    for rad, iteration, mean_strain, ptcld, strain in zip(rads, iterations, mean_strains, ptclds, strains):
        if not os.path.exists(os.path.join(save_path, scan_name)):
            os.mkdir(os.path.join(save_path, scan_name))

        # Create nx4 ndarray of ptcld and strain values
        strain_map = np.concatenate((np.asarray(ptcld.points), np.expand_dims(strain, axis=1)), axis=1)
        plot_thickness_array(strain_map, scan_name, save_location, f"strain_rad{rad}_iter{iteration}", "Strain")

    if not os.path.exists(save_file):
        with open(save_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header1)
            writer.writerow(header2)

    with open(save_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    return


def deficiency_data(scan_minus_prepost):
    """Returns the deficiency of the subject"""
    subjs = ["CSACL020", "CSACL022", "CSACL031", "CSACL036", "CSACL052", "CSACL054", "CSACL055", "CSACL057", "CSACL059",
             "CSACL060", "CSACL065", "CSACL066", "CSACL070"]
    deficients = ["L", "L", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "L"]

    limb = scan_minus_prepost[8]
    subj = scan_minus_prepost[0:8]
    ind = subjs.index(subj)

    if deficients[ind] == limb:
        return "ACL Deficient"
    else:
        return "Healthy Contralateral"


if __name__ == "__main__":
    # Options:
    predict_volumes_option = False  # Predict with network

    correct_volumes_option = False  # Use corrected segmentations
    derandomize_option = False  # Derandomize the corrected segmentations

    create_point_clouds_option = False

    # Geomagic here

    register_point_clouds_option = True
    visualize_registration_option = False
    visualize_strain_map_option = False

    # Declarations
    model_name = "unet_2024-07-11_00-40-25_ctHT5.h5"
    batch_size = 10
    n_slices = 80
    batches_per_scan = n_slices // batch_size
    excluded_list = ["CSACL065LT"]  # Scans excluded

    # Predict patella and patellar cartilage volumes
    if predict_volumes_option:
        # Load in dataset
        image_dir = get_data_path("Crook_ct") + os.sep + "*.bmp"
        dataset = get_paranjape_dataset(image_dir, batch_size=batch_size)
        iterable = iter(dataset)

        # Load in model
        model_filename = get_model_filename(model_name)
        model = load_model(model_filename)

        # Iterate through scans in dataset, make predictions, create volumes, save ndarrays
        for i in range(len(dataset) // batches_per_scan):

            for cur_batch_num in range(1, batches_per_scan + 1):
                filename, mri = next(iterable)
                pred_label = model.predict(mri)
                pat, pat_cart = process_predicted_label(pred_label)
                mri = reshape_mri(mri)
                # If we're predicting the first batch of images in the scan, assign volume to prediction,
                # and get scan name to save the volumes
                if cur_batch_num == 1:
                    mri_vol = mri
                    pat_vol = pat
                    pat_cart_vol = pat_cart
                    scan_name = parse_scan_name(filename)

                # If we're not predicting first batch, append to current batch
                elif cur_batch_num <= batches_per_scan:
                    mri_vol = np.concatenate((mri_vol, mri), axis=2)
                    pat_vol = np.concatenate((pat_vol, pat), axis=2)
                    pat_cart_vol = np.concatenate((pat_cart_vol, pat_cart), axis=2)

                # If we've predicted final batch for scan, save volume
                if cur_batch_num == batches_per_scan:
                    save_volumes(pat_vol, pat_cart_vol, model_name, scan_name)
                    save_bmps(mri_vol, pat_vol, pat_cart_vol, model_name, scan_name)
                    print(f"Saved {scan_name}, scan {i} of {(len(dataset) // batches_per_scan) - 1}")

    if correct_volumes_option:
        image_path = get_data_path("Crook_Predictions")

        # Derandomize corrections
        if derandomize_option:
            derandomize(image_path, model_name[0:-3])

        # Load in corrected images
        image_path = os.path.join(image_path, model_name[0:-3], "corrected")
        scans = os.listdir(image_path)

        # Iterate through each folder, and assemble volumes of patella and patellar cartilage
        for scan in scans:
            if scan_criteria(scan, excluded_list):
                pat_vol = np.zeros((256, 256, 80))
                pat_cart_vol = np.zeros((256, 256, 80))
                images = os.listdir(os.path.join(image_path, scan, "ML_pat"))
                for image in images:
                    # Read in image
                    pat = Image.open(os.path.join(image_path, scan, "ML_pat", image))
                    pat_cart = Image.open(os.path.join(image_path, scan, "ML_patcart", image))

                    # Convert to np array
                    pat = np.array(pat)
                    pat_cart = np.array(pat_cart)

                    # Get slice number
                    slice_num = int(image.split("-")[1].split(".")[0]) - 21

                    # Save to volume
                    pat_vol[:, :, slice_num] = pat
                    pat_cart_vol[:, :, slice_num] = pat_cart

                save_volumes(pat_vol, pat_cart_vol, model_name, scan)

    # Create point clouds of the patella, patellar cartilage, and articulating surface of patella
    if create_point_clouds_option:
        # Get volume data path (what we're reading in)
        volume_path = os.path.join(get_data_path("Crook_Volumes"), model_name[0:-3])
        scans = os.listdir(volume_path)

        # Create point cloud data path (what we're saving to)
        point_cloud_path = os.path.join(get_data_path("Crook_PCs"), model_name[0:-3])
        if not os.path.exists(point_cloud_path):
            os.mkdir(point_cloud_path)

        # Iterate through each volume, calculate coordinate arrays, and save
        for scan in scans:
            if scan_criteria(scan, excluded_list):
                pat_vol, pat_cart_vol = load_volumes(scan, volume_path)
                p_array, pc_array = get_coordinate_arrays(pat_vol, pat_cart_vol)
                save_coordinate_arrays(p_array, pc_array, scan)

    # Load pre and post, register, calculate strain map, save registered point clouds and strain map
    if register_point_clouds_option:
        # Get point cloud data path (what we're reading in)
        point_cloud_path = os.path.join(get_data_path("Crook_ToGeomagic"), "P")
        scans = os.listdir(point_cloud_path)
        scans_new = []
        for scan in scans:
            scans_new.append(scan[:-6])
        scans = scans_new

        # Get strain data path (what we're saving to)
        strain_path = os.path.join(get_data_path("Crook_PCs"), model_name[0:-3])
        if not os.path.exists(strain_path):
            os.mkdir(strain_path)

        # Initialize info
        info = {}
        info["Subject ID"] = []
        info["Group"] = []
        info["Mean Strain"] = []
        info["Change_in_Thickness"] = []

        save_location = "cropped_crook_3"

        # Iterate through each scan and take in a pair of scans
        for idx in range(len(scans)):
            if idx % 2 != 0:
                print(f"Current Scans: {scans[idx]} and {scans[idx-1]}")
                pre_p_array, pre_pc_array = load_coordinate_arrays(scans[idx])
                post_p_array, post_pc_array = load_coordinate_arrays(scans[idx-1])

                # Calculate thickness
                pre_pc_array = calculate_thickness(pre_p_array, pre_pc_array)
                post_pc_array = calculate_thickness(post_p_array, post_pc_array)

                # Plot thickness arrays
                plot_thickness_array(pre_pc_array, scans[idx], save_location, "thick_raw", "Thickness (mm)")
                plot_thickness_array(post_pc_array, scans[idx - 1], save_location, "thick_raw", "Thickness (mm)")

                # Extract thicknesses
                pre_thickness = pre_pc_array[:, 3]
                post_thickness = post_pc_array[:, 3]
                info["Change_in_Thickness"].append(np.mean(pre_thickness) - np.mean(post_thickness))
                info["Group"].append(deficiency_data(scans[idx][0:10]))

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
                strain_map, cropping_list = produce_strain_map(pre_pc_ptcld, pre_thickness, post_pc_ptcld, post_thickness, visualize_strain_map_option, save_location, scans[idx])
                # save_cropping_list(scans[idx][:-3], cropping_list, save_location)

                # Calculate the strain at the middle-most point of the strain map
                # strains = list(strain_map[:, 3])
                # plt.hist(strains)
                # plt.xlabel("Strain")
                # plt.show()

                mean_strain = np.min(strain_map[:, 3])
                print(f"Mean strain for {scans[idx]}: {mean_strain}")

                info = scan_properties(scans[idx], scans[idx - 1], info, mean_strain)

        output_plots(info)
