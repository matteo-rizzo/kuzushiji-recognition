import os
import shutil
import sys
from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

pred_out_w, pred_out_h = 128, 128


def draw_rectangle(box_and_score, img, color):
    number_of_rect = np.minimum(500, len(box_and_score))

    for i in reversed(list(range(number_of_rect))):
        top, left, bottom, right = box_and_score[i, :]

        top = np.floor(top + 0.5).astype('int32')
        left = np.floor(left + 0.5).astype('int32')
        bottom = np.floor(bottom + 0.5).astype('int32')
        right = np.floor(right + 0.5).astype('int32')

        draw = ImageDraw.Draw(img)

        thickness = 4
        if color == "red":
            rect_color = (255, 0, 0)
        elif color == "blue":
            rect_color = (0, 0, 255)
        else:
            rect_color = (0, 0, 0)

        if i == 0:
            thickness = 4
        for j in range(2 * thickness):  # Disegna diversi strati perchè è sottile
            draw.rectangle([left + j, top + j, right - j, bottom - j],
                           outline=rect_color)

    del draw
    return img


def check_iou_score(true_boxes, detected_boxes, iou_thresh):
    iou_all = []
    for detected_box in detected_boxes:
        y1 = np.maximum(detected_box[0], true_boxes[:, 0])
        x1 = np.maximum(detected_box[1], true_boxes[:, 1])
        y2 = np.minimum(detected_box[2], true_boxes[:, 2])
        x2 = np.minimum(detected_box[3], true_boxes[:, 3])

        cross_section = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)
        all_area = (detected_box[2] - detected_box[0]) * (detected_box[3] - detected_box[1]) + (
                true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        iou = np.max(cross_section / (all_area - cross_section))
        # argmax=np.argmax(cross_section/(all_area-cross_section))
        iou_all.append(iou)
    score = 2 * np.sum(iou_all) / (len(detected_boxes) + len(true_boxes))
    print("score:{}".format(np.round(score, 3)))
    return score


def user_check(save_dir):
    """
    Asks the user confirmation before deleting all files in the save dir

    :param save_dir: the path to the save dir
    """

    dir_exists = os.path.isdir(save_dir)
    all_files = os.listdir(save_dir)

    # If a directory already exists and it is not empty, ask the user what to do
    if dir_exists and len(all_files) > 0:
        user_input = input('WARNING! There seems to be some files in the folder in which '
                           'to save cropped characters.\n'
                           'Do you wish to delete all existing files and proceed with the operation?\n'
                           'Please refuse to abort the execution.\n'
                           'Confirm? [Y/n]\n')
        confirmations = ['y', 'Y', 'yes', 'ok']

        user_ok = True if user_input in confirmations else False

        if user_ok:
            # Remove directory and all its files
            shutil.rmtree(save_dir)
        else:
            # Exit and leave files untouched
            sys.exit(0)

    assert not dir_exists or len(all_files) == 0, 'Folder is not empty! Problem with deletion'

    # Create empty directory if there is not one
    if not dir_exists:
        os.mkdir(save_dir)


#################################################################


def get_bb_boxes(predictions: np.ndarray,
                 annotation_list: np.array,
                 print: bool = False) -> Dict[str, np.ndarray]:
    """
    Computes the bounding boxes and perform non maximum suppression

    :param predictions: array of predictions with shape (batch, out_width, out_height, n_cat + 4)
    :param annotation_list: list o samples where:
            - annotation_list[0] = path to image
            - annotation_list[1] = annotations, as np.array
            - annotation_list[2] = recommended height split
            - annotation_list[3] = recommended width split
    :param print: whether to show bboxes and iou scores
    :return: list of boxes, as [<image_path>, <category>, <score>, <ymin>, <xmin>, <ymax>, <xmax>].
            Note that <category> is always 0 in our case.
    """

    all_boxes = dict()

    for i in np.arange(0, predictions.shape[0]):
        image_path = annotation_list[i][0]
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        box_and_score = boxes_for_image(predicts=predictions[i],
                                        category_n=1,
                                        score_thresh=0.3,
                                        iou_thresh=0.4)
        # Bidimensional np.ndarray. Each row is (category,score,ymin,xmin,ymax,xmax)

        if len(box_and_score) == 0:
            continue

        true_boxes = annotation_list[i][1][:, 1:]  # c_x,c_y,width,height
        top = true_boxes[:, 1:2] - true_boxes[:, 3:4] / 2
        left = true_boxes[:, 0:1] - true_boxes[:, 2:3] / 2
        bottom = top + true_boxes[:, 3:4]
        right = left + true_boxes[:, 2:3]
        true_boxes = np.concatenate((top, left, bottom, right), axis=1)

        heatmap = predictions[i, :, :, 0]

        print_w, print_h = img.size
        # resize predicted box to original size. Leave unchanged score, category
        box_and_score = box_and_score * [1, 1, print_h / pred_out_h, print_w / pred_out_w,
                                         print_h / pred_out_h, print_w / pred_out_w]

        # Produce a dictionary { "image_path": np.ndarray([category,score,ymin,xmin,ymax,xmax]) }
        all_boxes[image_path] = box_and_score

        if print:
            check_iou_score(true_boxes, box_and_score[:, 2:], iou_thresh=0.5)
            img = draw_rectangle(box_and_score[:, 2:], img, "red")
            img = draw_rectangle(true_boxes, img, "blue")

            fig, axes = plt.subplots(1, 2, figsize=(15, 15))
            axes[0].imshow(img)
            axes[1].imshow(heatmap)
            plt.show()

    return all_boxes


# Originally NMS_all
def boxes_for_image(predicts, category_n, score_thresh, iou_thresh) -> np.ndarray:
    y_c = predicts[..., category_n] + np.arange(pred_out_h).reshape(-1, 1)
    x_c = predicts[..., category_n + 1] + np.arange(pred_out_w).reshape(1, -1)
    height = predicts[..., category_n + 2] * pred_out_h
    width = predicts[..., category_n + 3] * pred_out_w

    count = 0
    # In our case category_n = 1, so category=0 (just one cycle)
    for category in range(category_n):
        predict = predicts[..., category]
        mask = (predict > score_thresh)

        # If no center is predicted with enough confidence
        if not mask.all:
            continue
        box_and_score = boxes_image_nms(predict[mask], y_c[mask], x_c[mask], height[mask], width[mask],
                                        iou_thresh)
        box_and_score = np.insert(box_and_score, 0, category,
                                  axis=1)  # category,score,ymin,xmin,ymax,xmax
        if count == 0:
            box_and_score_all = box_and_score
        else:
            box_and_score_all = np.concatenate((box_and_score_all, box_and_score), axis=0)
        count += 1

    # Get indexes to sort by score descending order
    score_sort = np.argsort(box_and_score_all[:, 1])[::-1]
    box_and_score_all = box_and_score_all[score_sort]

    # If there are more than one box starting at same coordinate (ymin) remove it
    # So it keeps the one with the highest score
    _, unique_idx = np.unique(box_and_score_all[:, 2], return_index=True)
    # Sorted preserves original order of boxes
    return box_and_score_all[sorted(unique_idx)]


# Originally: NMS
def boxes_image_nms(score, y_c, x_c, height, width, iou_thresh, merge_mode=False):
    if merge_mode:
        score = score
        ymin = y_c
        xmin = x_c
        ymax = height
        xmax = width
    else:
        # --- Flattening ---
        score = score.reshape(-1)
        y_c = y_c.reshape(-1)
        x_c = x_c.reshape(-1)
        height = height.reshape(-1)
        width = width.reshape(-1)
        size = height * width

        xmin = x_c - width / 2  # left
        ymin = y_c - height / 2  # top
        xmax = x_c + width / 2  # right
        ymax = y_c + height / 2  # bottom

        inside_pic = (ymin > 0) * (xmin > 0) * (ymax < pred_out_h) * (xmax < pred_out_w)
        # outside_pic = len(inside_pic) - np.sum(inside_pic)

        normal_size = (size < (np.mean(size) * 10)) * (size > (np.mean(size) / 10))
        score = score[inside_pic * normal_size]
        ymin = ymin[inside_pic * normal_size]
        xmin = xmin[inside_pic * normal_size]
        ymax = ymax[inside_pic * normal_size]
        xmax = xmax[inside_pic * normal_size]

    # Sort boxes in descending order
    score_sort = np.argsort(score)[::-1]
    score = score[score_sort]
    ymin = ymin[score_sort]
    xmin = xmin[score_sort]
    ymax = ymax[score_sort]
    xmax = xmax[score_sort]

    area = ((ymax - ymin) * (xmax - xmin))

    boxes = np.concatenate((score.reshape(-1, 1),
                            ymin.reshape(-1, 1),
                            xmin.reshape(-1, 1),
                            ymax.reshape(-1, 1),
                            xmax.reshape(-1, 1)),
                           axis=1)

    # --- Non maximum suppression ---

    box_idx = np.arange(len(ymin))
    alive_box = []

    while len(box_idx) > 0:

        # Take the first index of the best bbox
        alive_box.append(box_idx[0])

        y1 = np.maximum(ymin[0], ymin)
        x1 = np.maximum(xmin[0], xmin)
        y2 = np.minimum(ymax[0], ymax)
        x2 = np.minimum(xmax[0], xmax)

        cross_h = np.maximum(0, y2 - y1)
        cross_w = np.maximum(0, x2 - x1)
        still_alive = (((cross_h * cross_w) / area[0]) < iou_thresh)
        if np.sum(still_alive) == len(box_idx):
            print("error")
            print(np.max((cross_h * cross_w)), area[0])
        ymin = ymin[still_alive]
        xmin = xmin[still_alive]
        ymax = ymax[still_alive]
        xmax = xmax[still_alive]
        area = area[still_alive]
        box_idx = box_idx[still_alive]

    return boxes[alive_box]  # score,top,left,bottom,right


def annotations_to_bounding_boxes(annotations: List) -> np.array:
    """
    Utility to convert between the annotation format, and the format required by
    get_crop_characters_train function for character

    :param annotations: List[str, np.array] with image path and image annotations where:
        - ann[:, 0] = class
        - ann[:, 1] = x_center
        - ann[:, 2] = y_center
        - ann[:, 3] = x width
        - ann[:, 4] = y height
    :return: Dict of {image_path: ndarray([char_class,ymin,xmin,ymax,xmax])}
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


def predictions_to_bounding_boxes(predictions: List[np.array]) -> List[np.array]:
    """
    Utility to convert from prediction output to input array of get_crop_characters_train

    :param predictions: list of boxes as predicted by the detection model:
            So list of [image_path,category,score,ymin,xmin,ymax,xmax]
    :return:
    """
    # FIXME: probably useless. We'll see.
    pass


def create_crop_character_test(images_to_split: Dict[str, np.array],
                               save_dir: str) -> List[str]:
    """
    Crop image into all its bounding boxes, saving a different image for each one in save_dir.
    :param images_to_split: dict of {image_path: ndarray([ymin,xmin,ymax,xmax])}
    :param save_dir: directory where to save cropped images
    """
    # TODO: test

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
    :param images_to_split: dict of {image_path: ndarray([char_class,ymin,xmin,ymax,xmax])}
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

    assert os.path.isdir(save_dir), "Error: save_dir doesn't exists!"

    csv_path = os.path.join(save_dir, 'crop_list.csv')

    if mode == 'train':
        assert os.path.isfile(csv_path), "Error: csv file 'crop_list.csv' doesn't exists in path {}".format(csv_path)

        csv_df = pd.read_csv(csv_path, delimiter=',')

        n_rows = len(csv_df.index)

        assert len(os.listdir(save_dir)) - 1 == n_rows, "Error: csv and save_dir contains different number of items"

        return [tuple(c) for c in csv_df.values]

    if mode == 'test':
        assert not os.path.isfile(csv_path), "Error: there is an unexpected csv file in save_dir at {}." \
            .format(save_dir)

        img = sorted(os.listdir(save_dir))

        assert len(img) > 0, 'Error: provided save directory {} is empty'.format(save_dir)

        return img

    raise ValueError("Mode value {} is not valid. Possibilities are 'test' or 'train'.".format(mode))
