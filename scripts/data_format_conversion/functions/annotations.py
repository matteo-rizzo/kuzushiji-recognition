import os

import pandas as pd
import regex as re
from PIL import Image

from scripts.data_format_conversion.functions.darkflow_conversion import convert_to_darkflow, write_as_darkflow
from scripts.data_format_conversion.functions.yolov2_conversion import convert_to_yolov2, write_as_yolov2
from scripts.utils.utils import to_file_name, to_id


def delete_annotations(path_to_annotations):
    """
    Deletes all the previous annotations.
    :param path_to_annotations: the path where the previously generated annotations are stored
    """

    file_list = [f for f in os.listdir(path_to_annotations)]

    if file_list:
        print('\nDeleting previously generated annotations at {}'.format(path_to_annotations))

        for f in file_list:
            os.remove(os.path.join(path_to_annotations, f))
    else:
        print('\nNo previously generated annotations to delete at {}'.format(path_to_annotations))


def get_annotation_data(image_base_name: str,
                        path_to_images: str,
                        ann_format: str,
                        label_mapping: pd.DataFrame,
                        class_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the annotation data related to the labels of the given image.

    :param image_base_name: the base name of the image (without extension)
    :param path_to_images: the path where the images relative to the dataset are stored
    :param ann_format: the annotation format, which can be either YOLOv2 or darkflow
    :param label_mapping: the image-labels mapping
    :param class_mapping: the class string to class number mapping
    :return: the annotation of the current image as a dataframe
    """

    # Get all the labels of the image as a string
    labels = label_mapping.loc[image_base_name, 'labels']

    # Convert the string of labels to list
    labels = [line[:-1] for line in re.findall(r"(?:\S*\s){5}", str(labels))]

    # Get the width and height of the image
    img_width, img_height = Image.open(os.path.join(path_to_images, to_file_name(image_base_name))).size

    convert_to = {
        'YOLOv2': convert_to_yolov2,
        'darkflow': convert_to_darkflow
    }

    # Create a list of lists to store the annotation data
    annotation_data = [convert_to[ann_format](label, class_mapping, img_width, img_height) for label in labels]

    # If the image has no labels, insert just image width and height
    if not annotation_data:
        annotation_data = [['', '', '', '', '', img_width, img_height]]

    data_format = {
        'YOLOv2': ['class', 'x_c', 'y_c', 'bb_width', 'bb_height'],
        'darkflow': ['class', 'xmin', 'ymin', 'xmax', 'ymax', 'img_width', 'img_height']
    }

    # Create a dataframe to store the whole annotation
    annotation = pd.DataFrame(annotation_data, columns=data_format[ann_format])

    return annotation


def generate_annotations(path_to_annotations, path_to_images, path_to_map, path_to_classes, ann_format):
    """
    Generates an annotation file for each image in the dataset.

    :param path_to_map: the path to the image-labels mapping
    :param path_to_annotations: the path where di annotations of each image must be stored
    :param path_to_images: the path where the images relative to the dataset are stored
    :param path_to_classes: the path to the classes (unicode character and translation)
    :param ann_format: defines the format of the annotations (YOLOv2 or Darkflow)
    """

    # If no images folder exists, an error occurs
    if not os.path.isdir(path_to_images):
        raise Exception('No images folder found at {}!'.format(path_to_images))

    print('Images are stored at {}.'.format(path_to_images))

    print('\nGenerating the {format} annotations at {path}...'.format(format=ann_format,
                                                                      path=path_to_annotations))

    # If no annotations folder exists
    if not os.path.isdir(path_to_annotations):
        # Create the annotations folder
        os.makedirs(path_to_annotations)
    else:
        # Delete previously generated annotations
        delete_annotations(path_to_annotations)

    # Get the image-labels mapping
    image_labels_map = pd.read_csv(path_to_map, index_col='image_id')

    # Get the class number to character mapping
    class_numbers = pd.read_csv(path_to_classes)

    print('\nStarting the generation of the annotations...')
    print('\n.............................................')

    # Iterate over the names of the images
    for image_name in list(os.listdir(path_to_images)):
        # Get the base name of the image (without file extension)
        image_id = to_id(image_name)

        print('\nGenerating annotations for image {}...'.format(image_id))

        # Get the data for the annotation of the image
        annotation = get_annotation_data(image_base_name=image_id,
                                         path_to_images=path_to_images,
                                         label_mapping=image_labels_map,
                                         class_mapping=class_numbers,
                                         ann_format=ann_format)

        # Print the first 5 rows of the annotation
        if not annotation.empty:
            print(annotation.head())

        write_as = {
            'YOLOv2': write_as_yolov2,
            'darkflow': write_as_darkflow
        }

        # Write the annotation on file
        write_as[ann_format](annotation, path_to_annotations, image_id)

    print('\n {n_ann}/{n_img} annotations have been generated successfully.'
          .format(n_ann=len(list(os.listdir(path_to_annotations))),
                  n_img=len(list(os.listdir(path_to_images)))))
