import os
import pandas as pd

from scripts.utils.utils import to_file_name


def convert_to_frcnn(label: [], image_path: str, **_) -> []:
    """
    Converts the label data to the darkflow data format (i.e. PASCAL VOC XML).

    The dataset is provided with labels in the format: <unicode> <x_bl> <y_bl> <abs_bb_width> <abs_bb_height>

    Where:
    - <unicode>        : is the class label as a unicode
    - <x_bl>           : is the bottom-left corner x coordinate of the bounding box
    - <y_bl>           : is the bottom-left corner y coordinate of the bounding box
    - <abs_bb_width>   : is the width of the bounding box
    - <abs_bb_height>  : is the height of the bounding box

    The darkflow data format is: <class_name> <xmin> <ymin> <xmax> <ymax> <img_width> <img_height>

    Where:
    - <class_name> : is the name of the class object as string
    - <img_width>  : is the width of the image
    - <img_height> : is the height of the image
    - <xmin>       : is the bottom-left x coordinate of the bounding box
    - <ymin>       : is the bottom-left y coordinate of the bounding box
    - <xmax>       : is the top-right x coordinate of the bounding box
    - <ymax>       : is the top-right y coordinate of the bounding box

    :param image_path: the path to the image, including image extension
    :param label: the label for one character in the image
    :return: a list with the converted data
    """

    # Get the label data in the dataset format
    unicode, xmin, ymin, abs_bb_width, abs_bb_height = label.split()

    # Make sure the class is a string
    class_name = str(unicode)

    # Cast each data to int
    xmin = int(xmin)
    ymin = int(ymin)
    abs_bb_width = int(abs_bb_width)
    abs_bb_height = int(abs_bb_height)

    # Calculate the top-right coordinates of the bounding box
    xmax = xmin + abs_bb_width
    ymax = ymin + abs_bb_height

    return [to_file_name(image_path, 'jpg'),
            xmin,
            ymin,
            xmax,
            ymax,
            class_name]


def write_as_frcnn(annotation: pd.DataFrame,
                   path_to_annotations: str,
                   image_id: str):
    """
    Writes an annotation on file according to the darkflow format.

    :param annotation: the annotation data
    :param path_to_annotations: the path to the annotations files
    :param image_id: the base name of the image (without file extension)
    """

    annotation.to_csv(os.path.join(path_to_annotations, '..', 'annotations.txt'),
                      header=None,
                      index=None,
                      mode='a',
                      sep=',')
