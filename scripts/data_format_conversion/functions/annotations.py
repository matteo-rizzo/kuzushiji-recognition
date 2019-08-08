import os
import pandas as pd
import regex as re

from PIL import Image
from pascal_voc_writer import Writer

from scripts.utils.utils import to_file_name, to_id


def write_annotation(annotation: pd.DataFrame,
                     path_to_annotations: str,
                     image_id: str,
                     ann_format: str):
    """
    Writes an annotation on file based on the selected format.

    :param annotation: the annotation data
    :param path_to_annotations: the path to the annotations files
    :param image_id: the base name of the image (without file extension)
    :param ann_format: the format of the annotation, which may be:
                                - YOLOv2
                                - darkflow
    """

    # Create the txt annotation file for the image
    if ann_format == 'YOLOv2':
        annotation.to_csv(os.path.join(path_to_annotations,
                                       to_file_name(image_id, 'txt')),
                          header=None,
                          index=None,
                          sep=' ',
                          mode='a')

    # Create the xml annotation file for the image
    if ann_format == 'darkflow':

        # Create a writer object for the image
        writer = Writer(os.path.join(path_to_annotations,
                                     to_file_name(image_id, 'jpg')),
                        width=800,
                        height=400)

        # Add the data related to each row (label) of the dataframe
        for _, row in annotation.iterrows():
            writer.addObject(name=row['name'],
                             xmin=row['xmin'],
                             ymin=row['ymin'],
                             xmax=row['xmax'],
                             ymax=row['ymax'])

        # Write the data to an XML file
        writer.save(os.path.join(path_to_annotations, to_file_name(image_id, 'xml')))


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


def convert_to_yolov2(label: [],
                      class_mapping: pd.DataFrame,
                      image_width: int,
                      image_height: int) -> []:
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

    # Cast each data to int
    xmin = int(xmin)
    ymin = int(ymin)
    abs_bb_width = int(abs_bb_width)
    abs_bb_height = int(abs_bb_height)

    # Calculate the top-right coordinates of the bounding box
    xmax = xmin + abs_bb_width
    ymax = ymin + abs_bb_height

    return [unicode,
            image_width,
            image_height,
            xmin,
            ymin,
            xmax,
            ymax]


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
    :return:
    """

    # Get all the labels of the image as a string
    labels = label_mapping.loc[image_base_name, 'labels']

    # Convert the string of labels to list
    labels = [line[:-1] for line in re.findall(r"(?:\S*\s){5}", str(labels))]

    # Get the width and height of the image
    img_width, img_height = Image.open(os.path.join(path_to_images, to_file_name(image_base_name))).size

    conversion = {
        'YOLOv2': convert_to_yolov2,
        'darkflow': convert_to_darkflow
    }

    # Create a list of lists to store the annotation data
    annotation_data = [conversion[ann_format](label, class_mapping, img_width, img_height) for label in labels]

    # Create a dataframe to store the whole annotation
    annotation = pd.DataFrame(annotation_data,
                              columns=['class',
                                       'x_c',
                                       'y_c',
                                       'bb_width',
                                       'bb_height'])

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

    print('\nGenerating the {format} annotations at {path}...\n'.format(format=ann_format,
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

    # Iterate over the names of the images
    for image_name in list(os.listdir(path_to_images)):
        image_id = to_id(image_name)

        print('Generating annotations for image {}...'.format(image_id))

        # Get the data for the annotation of the image
        annotation = get_annotation_data(image_base_name=image_id,
                                         path_to_images=path_to_images,
                                         label_mapping=image_labels_map,
                                         class_mapping=class_numbers,
                                         ann_format=ann_format)

        # Print the first 5 rows of the annotation
        print(annotation.head())

        # Write the annotation on file
        write_annotation(annotation, path_to_annotations, image_id, ann_format)

    print('\n {n_ann}/{n_img} annotations have been generated successfully.'
          .format(n_ann=len(list(os.listdir(path_to_annotations))),
                  n_img=list(os.listdir(path_to_images))))
