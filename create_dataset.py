import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np


def load_data(image_path, mask_path):

    # image_byte = image_path.numpy()
    # mask_byte = mask_path.numpy()

    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)

    image = tf.io.decode_image(image)
    mask = tf.io.decode_image(mask)

    # image = np.load(image_str)
    # image = image['arr']
    # image = image.astype(np.float64) / 255.0
    #
    # mask = np.load(mask_str)
    # mask = mask['arr']

    return image, mask


def create_dataset(image_dir, mask_dir):
    # Create dataset from list of image files
    mri = tf.data.Dataset.list_files(image_dir, shuffle=False)
    mask = tf.data.Dataset.list_files(mask_dir, shuffle=False)

    dataset = tf.data.Dataset.zip((mri, mask))

    dataset = dataset.map(load_data)

    return dataset


def get_dataset(batch_size, dataset_type):
    if dataset_type == 'train':
        images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri"
        masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d"
    elif dataset_type == 'test':
        images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/test/mri"
        masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/test/mask_3d"
    elif dataset_type == 'val' or dataset_type == 'validation':
        images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/val/mri"
        masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/val/mask_3d"
    else:
        raise(ValueError, f"The value {dataset_type} for the variable dataset_type is not one of 'train', 'test', "
                          f"or 'val'")

    # Add .npz extension
    images_dir = images_dir + "/*.npz"
    masks_dir = masks_dir + "/*.npz"

    # Create dataset
    dataset = create_dataset(images_dir, masks_dir)

    return dataset


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 32
    images_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri/*.npz"
    masks_dir = "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d/*.npz"

    # Create a dataset of file paths
    # image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    # label_files = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    dataset = create_dataset(images_dir, masks_dir)

    for mri, mask in dataset:
        print(mri, mask)
        break
