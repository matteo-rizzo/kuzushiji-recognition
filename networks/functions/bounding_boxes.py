from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from networks.functions.utils import check_iou_score, draw_rectangle

pred_out_w, pred_out_h = 128, 128


def get_bb_boxes(predictions: np.ndarray,
                 mode: str,
                 annotation_list: np.array = None,
                 test_images_path: List[str] = None,
                 print: bool = False) -> Dict[str, np.ndarray]:
    """
    Computes the bounding boxes and perform non maximum suppression

    :param mode: a string between 'test' and 'train'.
                In test mode no 'annotation_list' is provided, but instead must be provided
                'test_images_path'.
                In train mode the 'annotation_list' must be provided, but no 'test_images_path' will
                be considered.
    :param predictions: array of predictions with shape (batch, out_width, out_height, n_cat + 4)
    :param annotation_list: list o samples where:
            - annotation_list[0] = path to image
            - annotation_list[1] = annotations, as np.array
            - annotation_list[2] = recommended height split
            - annotation_list[3] = recommended width split
    :param test_images_path: list of filepaths to test images
    :param print: whether to show bboxes and iou scores. Iou scores are available only in train mode.
    :return: dict of boxes, as {<image_path>: [<category>, <score>, <ymin>, <xmin>, <ymax>, <xmax>]}.
            Note that <category> is always 0 in our case.
    """

    all_boxes = dict()

    for i in tqdm(np.arange(0, predictions.shape[0])):

        if mode == 'train':
            image_path = annotation_list[i][0]
        elif mode == 'test':
            image_path = test_images_path[0]
        else:
            raise ValueError('Error: unsupported mode {}'.format(mode))

        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        box_and_score = boxes_for_image(predicts=predictions[i],
                                        category_n=1,
                                        score_thresh=0.3,
                                        iou_thresh=0.4)
        # Bidimensional np.ndarray. Each row is (category,score,ymin,xmin,ymax,xmax)

        if len(box_and_score) == 0:
            continue

        heatmap = predictions[i, :, :, 0]

        print_w, print_h = img.size
        # resize predicted box to original size. Leave unchanged score, category
        box_and_score = box_and_score * [1, 1, print_h / pred_out_h, print_w / pred_out_w,
                                         print_h / pred_out_h, print_w / pred_out_w]

        # Produce a dictionary { "image_path": np.ndarray([category,score,ymin,xmin,ymax,xmax]) }
        all_boxes[image_path] = box_and_score

        if mode == 'train' and print:
            true_boxes = annotation_list[i][1][:, 1:]  # c_x,c_y,width,height
            top = true_boxes[:, 1:2] - true_boxes[:, 3:4] / 2
            left = true_boxes[:, 0:1] - true_boxes[:, 2:3] / 2
            bottom = top + true_boxes[:, 3:4]
            right = left + true_boxes[:, 2:3]
            true_boxes = np.concatenate((top, left, bottom, right), axis=1)

            check_iou_score(true_boxes, box_and_score[:, 2:], iou_thresh=0.5)
            img = draw_rectangle(box_and_score[:, 2:], img, "red")
            img = draw_rectangle(true_boxes, img, "blue")

            fig, axes = plt.subplots(1, 2, figsize=(15, 15))
            axes[0].imshow(img)
            axes[1].imshow(heatmap)
            plt.show()

        if mode == 'test' and print:
            img = draw_rectangle(box_and_score[:, 2:], img, "red")
            fig, axes = plt.subplots(1, 2, figsize=(15, 15))
            axes[0].imshow(img)
            axes[1].imshow(heatmap)
            plt.show()

    return all_boxes


def boxes_for_image(predicts, category_n, score_thresh, iou_thresh) -> np.ndarray:
    """
    Given all the bbox for a single image, returns all the non non-maximum suppress bboxes

    :param predicts: all the bbox of an image (128x128x5)
    :param category_n: the number of classes (1 in our case)
    :param score_thresh: the minimum confidence threshold for non-maximum suppression
    :param iou_thresh: the minimum IoU threshold for non-maximum suppression
    :return:
    """

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

        box_and_score = boxes_image_nms(predict[mask],
                                        y_c[mask],
                                        x_c[mask],
                                        height[mask],
                                        width[mask],
                                        iou_thresh)

        # Insert <category> into box_and_score (which has the structure <score> <ymin> <xmin> <ymax> <xmax>)
        box_and_score = np.insert(box_and_score,
                                  0,
                                  category,
                                  axis=1)
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


def boxes_image_nms(score, y_c, x_c, height, width, iou_thresh, merge_mode=False) -> np.array:
    """
    Performs the non-maximum suppression on the given bboxes

    :param score: the confidence score of the bbox (flatten array)
    :param y_c: the y coordinate of the center of the bbox (flatten array)
    :param x_c: the x coordinate of the center of the bbox (flatten array)
    :param height: the height of the bbox (flatten array)
    :param width: the width of the bbox (flatten array)
    :param iou_thresh: the minimum IoU threshold for the suppression
    :param merge_mode:
    :return: an array of bboxes with the following structure: <score> <top> <left> <bottom> <right>
    """

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

    return boxes[alive_box]
