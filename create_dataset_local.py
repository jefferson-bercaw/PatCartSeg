import tensorflow as tf
import os
import numpy as np


def load_npz_image(filepath):
    image = np.load(filepath)
    image = image['arr']
    tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor_image = tensor_image / 255.0
    return tensor_image


def load_npz_mask(filepath):
    image = np.load(filepath)
    image = image['arr']
    tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
    return tensor_image


if __name__ == '__main__':

    start_path = 'R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train'

    images_dir = 'R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mri'
    masks_dir = 'R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/mask_3d'

    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npz')]
    label_files = [os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npz')]

    mri = []
    masks = []
    counter = 1
    for image_path, mask_path in zip(image_files, label_files):
        # Load image and mask np arrays and convert to tf tensors
        image = load_npz_image(image_path)
        mask = load_npz_mask(mask_path)

        mri.append(image)
        masks.append(mask)

        if counter % 100 == 0:
            print(f"{counter} images completed")

        counter += 1

    # Create Tensors
    mri_tensor = tf.stack(mri)
    masks_tensor = tf.stack(masks)

    # Create Dataset
    dataset = tf.data.Dataset.from_tensor_slices((mri_tensor, masks_tensor))

    # Check Shape:
    for input_example, label_example in dataset.take(1):
        print("Input shape:", input_example.shape)
        print("Label shape:", label_example.shape)

    # Save Dataset
    tf.data.experimental.save(dataset, "R:/DefratePrivate/Bercaw/Patella_Autoseg/Split_Data/train/training_dataset.tfrecord")