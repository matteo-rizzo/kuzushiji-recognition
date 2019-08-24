import os
import xml.dom.minidom as minidom

import numpy as np
import pandas as pd

from scripts.utils.utils import to_file_name, draw_box_and_text, display_image


def visualize_bounding_boxes(path_to_images: str, image_id: str, labels: []):
    """
    Takes an image_id and return an image with bounding boxes around each character.

    :param path_to_images: the path to the images of the training set
    :param labels: the labels of the image to be tested
    :param image_id: an image base name (i.e. without the file extension)
    """

    plt = display_image(os.path.join(path_to_images, to_file_name(image_id)), show=False)
    ax = plt.gca()

    for label in labels:
        ax = draw_box_and_text(ax, label)

    plt.show()


def visualize_from_backup(path_to_images: str, path_to_mapping: str, image_id: str):
    """
    Visualizes the image from the original data.

    :param path_to_images: the path to the backup images
    :param path_to_mapping: the path to image-labels mapping
    :param image_id: the id of the image (without file extension)
    """

    print('\nRetrieving annotation from image_labels_map.csv...')

    mapping = pd.read_csv(path_to_mapping)
    mapping.set_index('image_id', inplace=True)

    labels = mapping.loc[image_id, 'labels']

    if not labels:
        raise Exception('Image {} has no characters hence no labels'.format(image_id))
    else:
        labels = np.array(labels.split(" ")).reshape(-1, 5)

    visualize_bounding_boxes(path_to_images, image_id, labels)


def visualize_from_training(path_to_images: str, path_to_annotations: str, image_id: str):
    """
    Visualizes the image from the current training set.

    :param path_to_images: the path to the backup images
    :param path_to_annotations: the path to the annotations
    :param image_id: the id of the image (without file extension)
    """

    print('\nRetrieving generated annotation...')

    xml_doc = minidom.parse(os.path.join(path_to_annotations, to_file_name(image_id, 'xml')))
    objects = xml_doc.getElementsByTagName('object')

    labels = []

    for obj in objects:
        obj_class = obj.getElementsByTagName('name')[0].firstChild.nodeValue

        bounding_box = obj.getElementsByTagName('bndbox')[0]
        xmin = bounding_box.getElementsByTagName('xmin')[0].firstChild.nodeValue
        ymin = bounding_box.getElementsByTagName('ymin')[0].firstChild.nodeValue
        xmax = bounding_box.getElementsByTagName('xmax')[0].firstChild.nodeValue
        ymax = bounding_box.getElementsByTagName('ymax')[0].firstChild.nodeValue

        label_data = [obj_class,
                      xmin,
                      ymin,
                      str(int(xmax) - int(xmin)),
                      str(int(ymax) - int(ymin))]

        labels.append(label_data)

    visualize_bounding_boxes(path_to_images, image_id, labels)
