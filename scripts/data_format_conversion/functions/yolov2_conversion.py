import os
import pandas as pd

from scripts.utils.utils import to_file_name


def convert_to_yolov2(label: [],
                      class_mapping: pd.DataFrame,
                      image_width: int,
                      image_height: int,
                      **_) -> []:  # **_ allows to pass indefinite number of irrelevant arguments
    """
    Converts the label data to the YOLO data format.

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

    :param label: the label for one character in the image
    :param class_mapping: the class number to character mapping
    :param image_width: the absolute the width of the image
    :param image_height: the absolute height of the image
    :return: a list with the converted data
    """

    # Get the label data in the dataset format
    unicode, x_bl, y_bl, abs_bb_width, abs_bb_height = label.split()

    # Cast each data to int
    x_bl = int(x_bl)
    y_bl = int(y_bl)
    abs_bb_width = int(abs_bb_width)
    abs_bb_height = int(abs_bb_height)

    # Convert the class string id to the corresponding integer value
    class_number = class_mapping[class_mapping.Unicode == unicode].index[0]

    # Calculate the normalized coordinates of the center of the bounding box
    x_c = (x_bl + abs_bb_width / 2) / image_width
    y_c = (y_bl + abs_bb_height / 2) / image_height

    # Calculate the normalized dimensions of the bounding box
    bb_width = abs_bb_width / image_width
    bb_height = abs_bb_height / image_height

    return [class_number,
            x_c,
            y_c,
            bb_width,
            bb_height]


def write_as_yolov2(annotation: pd.DataFrame,
                    path_to_annotations: str,
                    image_id: str):
    """
    Writes an annotation on file according to YOLOv2 format.

    :param annotation: the annotation data
    :param path_to_annotations: the path to the annotations files
    :param image_id: the base name of the image (without file extension)
    """

    annotation.to_csv(os.path.join(path_to_annotations, to_file_name(image_id, 'txt')),
                      header=None,
                      index=None,
                      sep=' ',
                      mode='a')
