import os
from xml.etree import ElementTree

import pandas as pd
from pascal_voc_writer import Writer

from scripts.utils.utils import to_file_name


def convert_to_darkflow(label: [],
                        class_mapping: pd.DataFrame,
                        image_width: int,
                        image_height: int) -> []:
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

    :param label: the label for one character in the image
    :param class_mapping: the class number to character mapping
    :param image_width: the absolute the width of the image
    :param image_height: the absolute height of the image
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

    return [class_name,
            image_width,
            image_height,
            xmin,
            ymin,
            xmax,
            ymax]


def write_as_darkflow(annotation: pd.DataFrame,
                      path_to_annotations: str,
                      image_id: str):
    """
    Writes an annotation on file according to the darkflow format.

    :param annotation: the annotation data
    :param path_to_annotations: the path to the annotations files
    :param image_id: the base name of the image (without file extension)
    """

    file_name = to_file_name(image_id, 'jpg').replace('-', '_')

    # Create a writer object for the image
    writer = Writer(path=os.path.join(path_to_annotations, file_name),
                    filename=os.path.join('..', 'images', file_name),
                    width=annotation['img_width'][0],
                    height=annotation['img_height'][0],
                    database='training')

    # Add the data related to each row (label) of the dataframe
    for _, row in annotation.iterrows():
        writer.addObject(name=row['class'],
                         xmin=row['xmin'],
                         ymin=row['ymin'],
                         xmax=row['xmax'],
                         ymax=row['ymax'])

    # Set the path to the XML annotation
    path_to_xml = os.path.join(path_to_annotations, to_file_name(image_id, 'xml'))

    # Write the data to an XML file
    writer.save(path_to_xml)

    # If the XML file is empty because the image has no objects
    if not annotation.loc[0]['class']:
        # Get the XML tree of the annotation
        tree = ElementTree.parse(path_to_xml)

        # Get the root of the annotation
        root = tree.getroot()

        # Delete the object tag
        obj = root.getchildren()[-1]
        root.remove(obj)
        ElementTree.dump(root)

        tree.write(path_to_xml)
