import os
import pandas as pd
from scripts.data_format_conversion.functions.utils import to_file_name, to_id, get_image_size, split_characters


def get_annotation_data(image_base_name: str,
                        path_to_images: str,
                        label_mapping: pd.DataFrame,
                        class_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the annotation data related to the labels of the given image.
    :param image_base_name: the base name of the image (without extension)
    :param path_to_images: the path where the images relative to the dataset are stored
    :param label_mapping: the image-labels mapping
    :param class_mapping: the class string to class number mapping
    :return:
    """

    # Create a dataframe to store the annotation
    annotation = pd.DataFrame(columns=['image_id', 'class', 'x_c', 'y_c', 'bb_width', 'bb_height'])

    # Get the width and height of the image
    image_width, image_height = get_image_size(os.path.join(path_to_images, to_file_name(image_base_name)))

    # Get all the labels of the image as a string
    all_labels = label_mapping.loc[image_base_name, 'labels']

    # Convert the string of labels to list
    all_labels = split_characters(all_labels)

    for label in all_labels:
        # Get the label data in the dataset format
        unicode, x_bl, y_bl, abs_bb_width, abs_bb_height = label.split()

        # Cast each data to int
        x_bl = int(x_bl)
        y_bl = int(y_bl)
        abs_bb_width = int(abs_bb_width)
        abs_bb_height = int(abs_bb_height)

        # Convert the class string id to the corresponding integer value
        class_number = class_mapping[class_mapping.Unicode == unicode].index

        # Calculate the normalized coordinates of the center of the bounding box
        x_c = (x_bl + abs_bb_width / 2) / image_width
        y_c = (y_bl + abs_bb_height / 2) / image_height

        # Calculate the normalized dimensions of the bounding box
        bb_width = abs_bb_width / image_width
        bb_height = abs_bb_height / image_height

        # Append the label to the annotation
        annotation.append([
            image_base_name,
            class_number,
            x_c,
            y_c,
            bb_width,
            bb_height
        ])

    return annotation


def generate_annotations(path_to_annotations, path_to_images, path_to_map, path_to_classes):
    """
    Generates an annotation file for each image in the dataset.

    The dataset is provided with labels in the format: <unicode> <x_bl> <y_bl> <abs_bb_width> <abs_bb_height>

    Where:
    - <unicode>        : is the class label as a unicode
    - <x_bl>           : is the bottom-left corner x coordinate of the bounding box
    - <y_bl>           : is the bottom-left corner y coordinate of the bounding box
    - <abs_bb_width>   : is the width of the bounding box
    - <abs_bb_height>  : is the height of the bounding box

    The YOLO format is: <class_number> <x_c> <y_c> <bb_width> <bb_height>.

    Where:
    - <class_number>            : an integer number from 0 to (classes-1) indicating the class of the object
    - <x_c> <y_c>               : the coordinates of the center of the rectangle (normalized between 0.0 and 1.0)
    - <bb_width> <bb_height>    : float values relative to width and height of the image for the bounding box
                                  (normalized between 0.0 and 1.0)

    The conversion is:

    - <class_number>    = mapping(<unicode>)
    - <x_c>             = <absolute_x_c>  / <image_width>
    - <y_c>             = <absolute_y_c>  / <image_height>
    - <bb_width>        = <abs_bb_width>  / <image_width>
    - <bb_height>       = <abs_bb_height> / <image_height>

    Where:
    - <absolute_x_c> = <x_bl> + <abs_bb_width>  / 2
    - <absolute_y_c> = <y_bl> + <abs_bb_height> / 2

    :param path_to_map: the path to the image-labels mapping
    :param path_to_annotations: the path where di annotations of each image must be stored
    :param path_to_images: the path where the images relative to the dataset are stored
    :param path_to_classes: the path to the classes (unicode character and translation)
    """

    print('Images are stored at {}.'.format(path_to_images))
    print('\nGenerating the annotations at {}...\n'.format(path_to_annotations))

    # If no annotations folder exists, it must be created
    if not os.path.isdir(path_to_images):
        raise Exception('No images folder found at {}!'.format(path_to_images))

    # If no images folder exists, an error occurs
    os.makedirs(path_to_annotations, exist_ok=True)

    # Get the image-labels mapping
    image_labels_map = pd.read_csv(path_to_map, index_col='image_id')

    class_numbers = pd.read_csv(path_to_classes)

    # Iterate over the names of the images
    for image_name in list(os.listdir(path_to_images)):
        print('Generating annotations for image {}...'.format(image_name))

        # Get the data for the annotation of the image
        annotation = get_annotation_data(image_base_name=to_id(image_name),
                                         path_to_images=path_to_images,
                                         label_mapping=image_labels_map,
                                         class_mapping=class_numbers)

        # Create the txt annotation file for the image
        annotation.to_csv(os.path.join(path_to_annotations,
                                       to_file_name(to_id(image_name), 'txt')),
                          header=None,
                          index=None,
                          sep=' ',
                          mode='a')

    print('\nAnnotations generated successfully!')
