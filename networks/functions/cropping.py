import os
import shutil
import sys
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def user_check(save_dir):
    """
    Asks the user confirmation before deleting all files in the save dir

    :param save_dir: the path to the save dir
    """

    # If a directory already exists and it is not empty, ask the user what to do
    if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
        user_input = input('WARNING! There seems to be some files in the folder in which '
                           'to save cropped characters.\n'
                           'The folder is {}\n'
                           'Do you wish to delete all existing files and proceed with the operation?\n'
                           'Please refuse to abort the execution.\n'
                           'Confirm? [Y/n]\n'.format(save_dir))
        confirmations = ['y', 'Y', 'yes', 'ok']

        user_ok = True if user_input in confirmations else False

        if user_ok:
            # Remove directory and all its files
            shutil.rmtree(save_dir)
        else:
            # Exit and leave files untouched
            sys.exit(0)

    assert not os.path.isdir(save_dir) or len(os.listdir(save_dir)) == 0, \
        'Folder is not empty! Problem with deletion'

    # Create empty directory if there is not one
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


def annotations_to_bounding_boxes(annotations: List[List]) -> Dict[str, np.array]:
    """
    Utility to convert between the annotation format, and the format required by
    create_crop_characters_train function for character

    :param annotations: List[str, np.array, ...] with image path and image annotations where:
        - ann[:, 0] = class
        - ann[:, 1] = x_center
        - ann[:, 2] = y_center
        - ann[:, 3] = x width
        - ann[:, 4] = y height
    :return: Dict of {image_path: ndarray([char_class, ymin, xmin, ymax, xmax])}
    """

    all_images_boxes = dict()

    # Take out unnecessary fields (keep only the first two)
    if len(annotations[0]) > 2:
        annotations = [(a[0], a[1]) for a in annotations]

    for img_path, ann in tqdm(annotations):
        ymin = ann[:, 2:3] - ann[:, 4:5] / 2  # y_center - height / 2
        xmin = ann[:, 1:2] - ann[:, 3:4] / 2  # x_center - width / 2
        ymax = ann[:, 2:3] + ann[:, 4:5] / 2  # y_center + height / 2
        xmax = ann[:, 1:2] + ann[:, 3:4] / 2  # x_center + width / 2

        assert ymin.shape == xmin.shape == ymax.shape == xmax.shape, 'Shape can\'t be different'

        # Create a column long as the number of bounding boxes and fill it with the image path
        # paths = np.full((ymin.shape[0], 1), img_path)

        dict_key = img_path
        dict_value = np.concatenate((ann[:, 0:1], ymin, xmin, ymax, xmax), axis=1)

        all_images_boxes[dict_key] = dict_value

    return all_images_boxes


def predictions_to_bounding_boxes(predictions: Dict[str, np.array]) -> Dict[str, np.array]:
    """
    Utility to convert from prediction output to input array of get_crop_characters_train

    :param predictions: list of boxes as predicted by the detection model. So a dict
                        {image: np.array[(category, score, xmin, ymin, xmax, ymax)]}
    :return: dict {image: np.array[xmin, ymin, xmax, ymax]}
    """

    assert predictions is not None, 'Predicted bounding box dictionary is None!\n Probably you forgot ' \
                                    'to set \'predict_on_test\' param to true for detection model.'

    result = dict()
    for k, v in tqdm(predictions.items()):
        result[k] = v[:, 2:]

    return result


def create_crop_characters_test(images_to_split: Dict[str, np.array],
                                save_dir: str) -> List[str]:
    """
    Crop image into all its bounding boxes, saving a different image for each one in save_dir.

    :param images_to_split: dict of {image_path: ndarray([ymin, xmin, ymax, xmax])}
    :param save_dir: directory where to save cropped images
    """
    # TODO: save on csv original images and their bounding box
    user_check(save_dir)

    # ---- Cropping ----

    cropped_list = []

    for img_path, boxes in tqdm(images_to_split.items()):

        # Get image name without extension, e.g. dataset/img.jpg -> img
        img_name = img_path.split(str(os.sep))[-1].split('.')[0]
        # Relative path with image name (no extension)
        img_name_path = os.path.join(save_dir, img_name)

        with Image.open(img_path) as img:
            # Give incremental id to each cropped box in image filename
            box_n = 0

            for box in boxes:
                ymin = float(box[0])  # top
                xmin = float(box[1])  # left
                ymax = float(box[2])  # bottom
                xmax = float(box[3])  # right

                filepath = img_name_path + '_' + str(box_n) + '.jpg'

                img.crop((xmin, ymin, xmax, ymax)).save(filepath)

                cropped_list.append(filepath)
                box_n += 1

    return cropped_list


def create_crop_characters_train(images_to_split: Dict[str, np.array],
                                 save_dir: str,
                                 save_csv: bool = True) -> List[Tuple[str, int]]:
    """
    Crops image into all bounding box, saving a different image for each one in save_dir.
    Additionally save a csv containing all pairs (char_image, char_class) in save_dir folder.

    :param save_csv: whether to save (cropped_img_path, char_class) to a cvs file
    :param images_to_split: dict of {image_path: ndarray([char_class, ymin, xmin, ymax, xmax])}
    :param save_dir: directory where to save cropped images
    """

    user_check(save_dir)

    # ---- Cropping ----

    cropped_list = []

    for img_path, boxes in tqdm(images_to_split.items()):

        # Get image name without extension, e.g. dataset/img.jpg -> img
        img_name = img_path.split(str(os.sep))[-1].split('.')[0]

        # Relative path with image name (no extension)
        img_name_path = os.path.join(save_dir, img_name)

        with Image.open(img_path) as img:
            # Give incremental id to each cropped box in image filename
            box_n = 0

            for box in boxes:
                char_class = int(box[0])
                ymin = float(box[1])  # top
                xmin = float(box[2])  # left
                ymax = float(box[3])  # bottom
                xmax = float(box[4])  # right

                filepath = img_name_path + '_' + str(box_n) + '.jpg'

                img.crop((xmin, ymin, xmax, ymax)).save(filepath)

                cropped_list.append((filepath, char_class))
                box_n += 1

    # Save list to csv, to rapidly load them
    if save_csv:
        csv_path = os.path.join(save_dir, 'crop_list.csv')
        df = pd.DataFrame(cropped_list, columns=['char_image', 'char_class'])
        df.to_csv(csv_path, sep=',', index=False)

    return cropped_list


def load_crop_characters(save_dir: str, mode: str) -> Union[List[Tuple[str, int]], List[str]]:
    """
    Loads the list of characters from file system. Useful to avoid regenerating cropped characters every time.

    :param mode: strings 'train' or 'test':
        - 'test': returned list will be a list of file paths to cropped images.
        - 'train' returned list will be composed of tuples (img_path, char_class)
    :param save_dir: file path which to search for objects in
    :return: list of character images whose format depending on 'mode' param
    """

    assert os.path.isdir(save_dir), "Error: save_dir doesn't exists at {}".format(save_dir)

    csv_path = os.path.join(save_dir, 'crop_list.csv')

    if mode == 'train':
        assert os.path.isfile(
            csv_path), "Error: csv file 'crop_list.csv' doesn't exists in path {}".format(csv_path)

        csv_df = pd.read_csv(csv_path, delimiter=',')

        n_rows = len(csv_df.index)

        assert len(os.listdir(
            save_dir)) - 1 == n_rows, "Error: csv and save_dir contains different number of items"

        return [tuple(c) for c in csv_df.values]

    if mode == 'test':
        assert not os.path.isfile(csv_path), "Error: there is an unexpected csv file in save_dir at {}." \
            .format(save_dir)

        img = sorted(os.listdir(save_dir))

        # Add relative path to image name
        img = [str(os.path.join(save_dir, name)) for name in img]

        assert len(img) > 0, 'Error: provided save directory {} is empty'.format(save_dir)

        return img

    raise ValueError("Mode value {} is not valid. Possibilities are 'test' or 'train'.".format(mode))
