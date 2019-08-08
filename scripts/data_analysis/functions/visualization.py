import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from scripts.utils.utils import to_file_name, draw_box_and_text, display_image


def visualize_random_image(path_to_train_images: str):
    """
    Visualizes a random image from the training set
    :param path_to_train_images: the path to the images of the training set
    """

    # Get a random index for the selection of the image
    rnd_nbr = random.randint(0, len(os.listdir(path_to_train_images)))

    # Get the random image base name
    rnd_img = os.listdir(path_to_train_images)[rnd_nbr]

    # Get the full path to the random image
    path_to_rnd_image = os.path.join(path_to_train_images, to_file_name(rnd_img))

    print('\nDisplaying the random image {}...'.format(rnd_img))

    # Display the random image
    display_image(path_to_rnd_image)


def visualize_null_images(path_to_train_images: str, mapping: pd.DataFrame):
    """
    Visualizes some of the images which have no characters.
    :param path_to_train_images: the path to the images of the training set
    :param mapping: the image-labels mapping
    """

    # Configure the plot
    rows, columns = 2, 2
    fig = plt.figure(figsize=(20, 20))

    # Get all the images with no characters
    images_nan_labels = mapping[mapping.isna().labels]['image_id'].tolist()

    # Plot some of the images with no characters
    for i in range(1, rows * columns + 1):
        rnd_img_nan_nbr = random.randint(0, len(images_nan_labels) - 1)
        img_nan = Image.open(os.path.join(
            path_to_train_images,
            to_file_name(images_nan_labels[rnd_img_nan_nbr])
        ))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img_nan, aspect='equal')

    print('\nDisplaying some images with no characters...')

    plt.show()


def visualize_training_data(path_to_train_images: str, image_id: str, mapping: pd.DataFrame):
    """
    Takes an image_id and return an image with bounding boxes around each character
    :param path_to_train_images: the path to the images of the training set
    :param mapping: the image-labels mapping
    :param image_id: an image base name (i.e. without the file extension)
    """

    print('\nDisplaying the image {} from the training set with bounding boxes...'.format(image_id))

    # Get all the characters and the position of the bounding boxes for the image
    mapping.set_index('image_id', inplace=True)
    labels = mapping.loc[image_id, 'labels']
    labels = np.array(labels.split(" ")).reshape(-1, 5)

    plt = display_image(os.path.join(path_to_train_images, to_file_name(image_id)), show=False)
    ax = plt.gca()

    for label in labels:
        ax = draw_box_and_text(ax, label)

    plt.show()


def visualize_images(path_to_dataset: str, path_to_train_images: str):
    """
    Visualize some images from the dataset.
    :param path_to_dataset: the path to the kaggle dataset folder
    :param path_to_train_images: the path to the images used for training
    """

    # Read the image-labels mapping csv file
    mapping = pd.read_csv(os.path.join(path_to_dataset, 'image_labels_map.csv'))

    # Visualize a random image
    visualize_random_image(path_to_train_images)

    # Visualize some images with no characters
    visualize_null_images(path_to_train_images, mapping)

    # Visualize the training data with bounding boxes
    test_img_id = '100241706_00004_2'
    visualize_training_data(path_to_train_images=path_to_train_images,
                            image_id=test_img_id,
                            mapping=mapping)
