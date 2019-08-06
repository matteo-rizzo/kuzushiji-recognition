import os
import pandas as pd
import regex as re
from PIL import Image


def split_characters(string):
    string = str(string)
    string = (re.findall(r"(?:\S*\s){5}", string))
    return [line[:-1] for line in string]


def generate_annotations(path_to_annotations, path_to_images, path_to_map):
    """
    Generate an annotation file for each image in the dataset.
    :param path_to_map: the path to the image-labels mapping
    :param path_to_annotations: the path where di annotations of each image must be stored
    :param path_to_images: the path where the images relative to the dataset are stored
    """

    print('Generating the annotations...')

    # If no annotations folder exists, it must be created
    if not os.path.isdir(path_to_images):
        raise Exception('No images folder found!')

    # If no images folder exists, an error occurs
    os.makedirs(path_to_annotations, exist_ok=True)

    # Get the image-labels mapping
    image_labels_map = pd.read_csv(path_to_map, index_col='image_id')

    print(image_labels_map)

    # Iterate over the names of the images
    for image_name in list(os.listdir(path_to_images)):
        # Take only the base name of the image without extension
        image_base_name = str(image_name.split('.')[0])

        # Create a dataframe to store the annotation
        annotation = pd.DataFrame(columns=['image_id', 'class', 'x', 'y', 'width', 'height'])

        # Get the width and height of the image
        image = Image.open(os.path.join(path_to_images, image_name))
        width, height = image.size

        # Get all the labels of the image as a string
        labels = image_labels_map.loc[image_base_name, 'labels']
        characters = split_characters(labels)

        # Create the full path to the annotation for the current image
        path_to_annotation = path_to_annotations + image_base_name + '.txt'

        # Create the annotation file for the current image
        annotation.to_csv(path_to_annotation, header=None, index=None, sep=' ', mode='a')
