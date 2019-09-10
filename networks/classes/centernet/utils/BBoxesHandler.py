from statistics import mean
from typing import Dict, Tuple
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
from tensorflow.python.keras.models import Model


class BBoxesHandler:

    def __init__(self, out_w: int = 128, out_h: int = 128, in_w: int = 512, in_h: int = 512):
        self.__pred_out_w = out_w
        self.__pred_out_h = out_h
        self.__pred_in_w = in_w
        self.__pred_in_h = in_h

    def __show_train_standard_bboxes(self, true_bboxes, predicted_bboxes, img, heatmap):
        # Draw true and predicted bboxes
        img = self.__draw_rectangle(predicted_bboxes, img, "red")
        img = self.__draw_rectangle(true_bboxes, img, "blue")

        # Plot the heatmap alongside the image with the bboxes
        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        axes[0].axis('off')
        axes[0].imshow(img)
        axes[1].axis('off')
        axes[1].imshow(heatmap)

        plt.show()

    def __show_test_standard_bboxes(self, predicted_bboxes, img, heatmap):
        # Draw predicted bboxes
        img = self.__draw_rectangle(predicted_bboxes, img, "red")

        # Plot the predicted heatmap alongside the image with the bboxes
        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        axes[0].imshow(img)
        axes[1].imshow(heatmap)

        plt.show()

    def get_train_standard_bboxes(self,
                                  predictions: np.ndarray,
                                  annotation_list: np.array = None,
                                  show: bool = False) -> Tuple[Dict[str, np.ndarray], float]:

        all_boxes = {}
        iou_scores = []

        for i in tqdm(np.arange(0, predictions.shape[0])):

            image_path = annotation_list[i][0]
            img = Image.open(image_path).convert("RGB")

            # Bidimensional np.ndarray. Each row is <score> <ymin> <xmin> <ymax> <xmax>
            bbox_and_score = self.__get_img_bboxes(predicts=predictions[i],
                                                   category_n=1,
                                                   score_thresh=0.3,
                                                   iou_thresh=0.4)

            if len(bbox_and_score) > 0:
                # Get width and height of the image
                print_w, print_h = img.size

                # Resize predicted box to original size. Leave unchanged score
                bbox_and_score = bbox_and_score * [1,
                                                   print_h / self.__pred_out_h,
                                                   print_w / self.__pred_out_w,
                                                   print_h / self.__pred_out_h,
                                                   print_w / self.__pred_out_w]

                # Produce a dictionary { "image_path": np.ndarray([score, ymin, xmin, ymax, xmax]) }
                all_boxes[image_path] = bbox_and_score

                # Create the true and predicted bboxes
                true_bboxes, predicted_bboxes = self.__get_true_and_pred_bboxes(annotation_list[i], bbox_and_score)

                # Append the IoU score of the current predictions to a list
                iou_score = self.__get_iou_score(true_bboxes, predicted_bboxes)
                print(image_path.split('/')[-1] + ' IoU score: {}'.format(iou_score))
                iou_scores.append(iou_score)

                if show:
                    self.__show_train_standard_bboxes(true_bboxes=true_bboxes,
                                                      predicted_bboxes=predicted_bboxes,
                                                      img=img,
                                                      heatmap=predictions[i, :, :, 0])
            else:
                continue

        return all_boxes, mean(iou_scores)

    def get_test_standard_bboxes(self,
                                 predictions: np.ndarray,
                                 test_images_path: List[str] = None,
                                 show: bool = False) -> Dict[str, np.ndarray]:

        all_boxes = {}

        for i in tqdm(np.arange(0, predictions.shape[0])):

            image_path = test_images_path[i]
            img = Image.open(image_path).convert("RGB")

            # Bidimensional np.ndarray. Each row is (score, ymin, xmin, ymax, xmax)
            # We removed category (from the original implementation)
            bbox_and_score = self.__get_img_bboxes(predicts=predictions[i],
                                                   category_n=1,
                                                   score_thresh=0.3,
                                                   iou_thresh=0.4)

            if len(bbox_and_score) >= 0:

                print_w, print_h = img.size

                # Resize predicted box to original size. Leave unchanged score
                bbox_and_score = bbox_and_score * [1,
                                                   print_h / self.__pred_out_h,
                                                   print_w / self.__pred_out_w,
                                                   print_h / self.__pred_out_h,
                                                   print_w / self.__pred_out_w]

                # Produce a dictionary { "image_path": np.ndarray([score, ymin, xmin, ymax, xmax]) }
                all_boxes[image_path] = bbox_and_score

                if show:
                    self.__show_test_standard_bboxes(predicted_bboxes=bbox_and_score[:, 1:],
                                                     img=img,
                                                     heatmap=predictions[i, :, :, 0])
            else:
                continue

        return all_boxes

    def __get_all_tiled_bboxes(self, offsets, k, pic, model):

        top_offsets, bottom_offsets, left_offsets, right_offsets = offsets
        k_w, k_h = k

        all_tile_boxes = np.array([])

        for top_offset, bottom_offset in zip(top_offsets, bottom_offsets):
            for left_offset, right_offset in zip(left_offsets, right_offsets):
                tile = cv2.resize(pic[top_offset:bottom_offset, left_offset:right_offset, :],
                                  (self.__pred_in_h, self.__pred_in_w))

                tile = np.array(tile, dtype=np.float32)
                tile /= 255

                # Output shape is (1, 128, 128, 5)
                predictions = model.predict(tile.reshape((1, self.__pred_in_h, self.__pred_in_w, 3)),
                                            batch_size=1,
                                            steps=1)

                # Boxes have format: <category> <score> <top> <left> <bot> <right>
                boxes = self.__get_img_bboxes(predictions, 1, 0.3, 0.4)

                if len(boxes) == 0:
                    continue

                # Reshape and add the offset
                boxes = boxes * [1,
                                 k_h / self.__pred_out_h,
                                 k_w / self.__pred_out_w,
                                 k_h / self.__pred_out_h,
                                 k_w / self.__pred_out_w] \
                        + np.array([0, top_offset, left_offset, top_offset, left_offset])

                if all_tile_boxes.size == 0:
                    all_tile_boxes = boxes
                else:
                    all_tile_boxes = np.concatenate((all_tile_boxes, boxes), axis=0)

        return all_tile_boxes

    def __show_test_tiled_bboxes(self, predicted_bboxes, image_path):

        image_show = Image.open(image_path).convert("RGB")
        image_show = self.__draw_rectangle(predicted_bboxes, image_show, "red")
        f, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.axis('off')
        plt.imshow(image_show)
        plt.show()

    def __show_train_tiled_bboxes(self, true_bboxes, predicted_bboxes, image_path):

        image_show = Image.open(image_path).convert("RGB")

        # Draw true and predicted bboxes
        image_show = self.__draw_rectangle(predicted_bboxes, image_show, "red")
        image_show = self.__draw_rectangle(true_bboxes, image_show, "blue")

        f, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.axis('off')
        plt.imshow(image_show)
        plt.show()

    @staticmethod
    def __get_true_and_pred_bboxes(example, bbox_and_score):

        # Boxes in the format: c_x, c_y, width, height
        true_bboxes = example[1][:, 1:]

        # Get the coords of the true bboxes
        top = true_bboxes[:, 1:2] - true_bboxes[:, 3:4] / 2
        left = true_bboxes[:, 0:1] - true_bboxes[:, 2:3] / 2
        bottom = top + true_bboxes[:, 3:4]
        right = left + true_bboxes[:, 2:3]

        # Create the true bboxes
        true_bboxes = np.concatenate((top, left, bottom, right), axis=1)

        # Get the predicted bboxes
        predicted_bboxes = bbox_and_score[:, 1:]

        return true_bboxes, predicted_bboxes

    def get_train_tiled_bboxes(self, dataset: np.array, model: Model, n_tiles: int, show: bool):
        """
        This functions behaves similarly to get_standard_bboxes, but it compute bboxes directly
         from each image using tiling to improve accuracy.

        :param dataset: the dataset in the format [[image path, annotations, height split, width split]]
        :param model: a trained keras model
        :param n_tiles: number of tiles t split the image
        :param show: whether to show results and scores
        :return:
        """

        all_boxes = {}
        iou_scores = []

        for example in tqdm(dataset):

            image_path = example[0]

            # Process the single image
            pic = Image.open(image_path)

            img_w, img_h = pic.size
            k_h = (4 * img_h) // (3 * n_tiles + 1)
            k_w = (4 * img_w) // (3 * n_tiles + 1)

            left_offsets = [int(i * 0.75 * k_w) for i in range(0, n_tiles)]
            right_offsets = [(img_w - lo) for lo in reversed(left_offsets)]
            top_offsets = [int(i * 0.75 * k_h) for i in range(0, n_tiles)]
            bottom_offsets = [(img_h - to) for to in reversed(top_offsets)]

            all_tile_boxes = self.__get_all_tiled_bboxes(
                offsets=(top_offsets, bottom_offsets, left_offsets, right_offsets),
                k=(k_w, k_h),
                pic=np.asarray(pic.convert('RGB'), dtype=np.uint8),
                model=model)

            if all_tile_boxes.size == 0:
                continue

            assert all_tile_boxes.ndim == 2, \
                'Error: numpy dimension must be 2, not {}'.format(all_tile_boxes.ndim)

            # bbox_and_score has structure: [<score> <ymin> <xmin> <ymax> <xmax>]
            bbox_and_score = self.__get_nms_bboxes(all_tile_boxes[:, 0],
                                                   all_tile_boxes[:, 1],
                                                   all_tile_boxes[:, 2],
                                                   all_tile_boxes[:, 3],
                                                   all_tile_boxes[:, 4],
                                                   iou_thresh=0.4,
                                                   tiled_mode=True)

            if bbox_and_score.size == 0:
                print('No boxes found')
                continue

            # Produce a dictionary { "image_path": np.ndarray([score, ymin, xmin, ymax, xmax]) }
            all_boxes[image_path] = bbox_and_score

            # Create the true and predicted bboxes
            true_bboxes, predicted_bboxes = self.__get_true_and_pred_bboxes(example, bbox_and_score)

            # Append the IoU score of the current predictions to a list
            iou_score = self.__get_iou_score(true_bboxes, predicted_bboxes)
            print(image_path.split('/')[-1] + ' IoU score: {}'.format(iou_score))
            iou_scores.append(iou_score)

            if show:
                self.__show_train_tiled_bboxes(true_bboxes, predicted_bboxes, image_path)

        # Return the dict and the mean of the IoU scores
        return all_boxes, mean(iou_scores)

    def get_test_tiled_bboxes(self, dataset: np.array, model: Model, n_tiles: int, show: bool):
        """
        This functions behaves similarly to get_standard_bboxes, but it compute bboxes directly
         from each image using tiling to improve accuracy.

        :param dataset: the dataset in the format [[image path, annotations, height split, width split]]
        :param model: a trained keras model
        :param n_tiles: number of tiles t split the image
        :param show: whether to show results and scores
        :return:
        """

        all_boxes = {}

        for example in tqdm(dataset):

            image_path = example

            # Process the single image
            pic = Image.open(image_path)

            img_w, img_h = pic.size
            k_h = (4 * img_h) // (3 * n_tiles + 1)
            k_w = (4 * img_w) // (3 * n_tiles + 1)

            left_offsets = [int(i * 0.75 * k_w) for i in range(0, n_tiles)]
            right_offsets = [(img_w - lo) for lo in reversed(left_offsets)]
            top_offsets = [int(i * 0.75 * k_h) for i in range(0, n_tiles)]
            bottom_offsets = [(img_h - to) for to in reversed(top_offsets)]

            all_tile_boxes = self.__get_all_tiled_bboxes(
                offsets=(top_offsets, bottom_offsets, left_offsets, right_offsets),
                k=(k_w, k_h),
                pic=np.asarray(pic.convert('RGB'), dtype=np.uint8),
                model=model)

            if all_tile_boxes.size == 0:
                continue

            assert all_tile_boxes.ndim == 2, \
                'Error: numpy dimension must be 2, not {}'.format(all_tile_boxes.ndim)

            # bbox_and_score has structure: [<score> <ymin> <xmin> <ymax> <xmax>]
            bbox_and_score = self.__get_nms_bboxes(all_tile_boxes[:, 0],
                                                   all_tile_boxes[:, 1],
                                                   all_tile_boxes[:, 2],
                                                   all_tile_boxes[:, 3],
                                                   all_tile_boxes[:, 4],
                                                   iou_thresh=0.4,
                                                   tiled_mode=True)

            if bbox_and_score.size == 0:
                print('No boxes found')
                continue

            # Produce a dictionary { "image_path": np.ndarray([score, ymin, xmin, ymax, xmax]) }
            all_boxes[image_path] = bbox_and_score

            if show:
                self.__show_test_tiled_bboxes(predicted_bboxes=bbox_and_score[:, 1:], image_path=image_path)

        # Return the dict
        return all_boxes

    def __get_img_bboxes(self, predicts, category_n, score_thresh, iou_thresh) -> np.ndarray:
        """
        Given all the bbox for a single image, returns all the non non-maximum suppress bboxes

        :param predicts: all the bbox of an image (128x128x5)
        :param category_n: the number of classes (1 in our case)
        :param score_thresh: the minimum confidence threshold for non-maximum suppression
        :param iou_thresh: the minimum IoU threshold for non-maximum suppression
        :return:
        """

        y_c = predicts[..., category_n] + np.arange(self.__pred_out_h).reshape(-1, 1)
        x_c = predicts[..., category_n + 1] + np.arange(self.__pred_out_w).reshape(1, -1)
        height = predicts[..., category_n + 2] * self.__pred_out_h
        width = predicts[..., category_n + 3] * self.__pred_out_w

        bbox_and_score = np.array([])

        # In our case category_n = 1, so category=0 (just one cycle)
        for category in range(category_n):
            predict = predicts[..., category]
            mask = (predict > score_thresh)

            # If no center is predicted with enough confidence
            if not mask.all:
                continue

            bbox_and_score = self.__get_nms_bboxes(predict[mask],
                                                   y_c[mask],
                                                   x_c[mask],
                                                   height[mask],
                                                   width[mask],
                                                   iou_thresh)

            # bbox_and_score has structure: <score> <ymin> <xmin> <ymax> <xmax>

            if bbox_and_score.size == 0:
                bbox_and_score = bbox_and_score
            else:
                bbox_and_score = np.concatenate((bbox_and_score, bbox_and_score), axis=0)

        # Get indexes to sort by score in descending order
        score_sort = np.argsort(bbox_and_score[:, 0])[::-1]
        bbox_and_score = bbox_and_score[score_sort]

        # If there are more than one box starting at same coordinate (ymin) remove it
        # So it keeps the one with the highest score
        _, unique_idx = np.unique(bbox_and_score[:, 1], return_index=True)

        # Sorted preserves original order of boxes
        return bbox_and_score[sorted(unique_idx)]

    @staticmethod
    def __flatten_bbox_data(score, y_c, x_c, height, width):
        score = score.reshape(-1)
        y_c = y_c.reshape(-1)
        x_c = x_c.reshape(-1)
        height = height.reshape(-1)
        width = width.reshape(-1)
        size = height * width

        return score, (x_c - width / 2, y_c - height / 2, x_c + width / 2, y_c + height / 2), size

    def __take_boxes_inside_image(self, xmin, xmax, ymin, ymax, size, score):
        inside_pic = (ymin > 0) * (xmin > 0) * (ymax < self.__pred_out_h) * (xmax < self.__pred_out_w)

        # Take only boxes of reasonable size
        normal_size = (size < (np.mean(size) * 10)) * (size > (np.mean(size) / 10))
        score = score[inside_pic * normal_size]
        ymin = ymin[inside_pic * normal_size]
        xmin = xmin[inside_pic * normal_size]
        ymax = ymax[inside_pic * normal_size]
        xmax = xmax[inside_pic * normal_size]

        return xmin, xmax, ymin, ymax, score

    @staticmethod
    def __nms(xmin, xmax, ymin, ymax, iou_thresh):
        box_idx = np.arange(len(ymin))
        boxes_to_keep = []

        # Box area of score-sorted boxes
        area = ((ymax - ymin) * (xmax - xmin))

        while len(box_idx) > 0:
            # Insert the index of the best bbox in the list of boxes to keep
            boxes_to_keep.append(box_idx[0])

            # y2 - y1 is an array with the distance (element-wise) between boxes on y coord
            y1 = np.maximum(ymin[0], ymin)
            x1 = np.maximum(xmin[0], xmin)
            y2 = np.minimum(ymax[0], ymax)
            x2 = np.minimum(xmax[0], xmax)

            # Bbox-wise area
            cross_h = np.maximum(0, y2 - y1)
            cross_w = np.maximum(0, x2 - x1)

            # Mask to keep just the boxes which overlap with best box for less than threshold
            max_over_lap = (((cross_h * cross_w) / area[0]) < iou_thresh)

            # Mask to keep just the boxes with enough out_lap surface
            min_out_lap = (((area - (cross_h * cross_w)) / area) > (1 - iou_thresh))

            still_alive = max_over_lap * min_out_lap

            assert np.sum(still_alive) != len(box_idx), \
                'An error occurred: {} {}'.format(np.max((cross_h * cross_w)), area[0])

            ymin = ymin[still_alive]
            xmin = xmin[still_alive]
            ymax = ymax[still_alive]
            xmax = xmax[still_alive]
            area = area[still_alive]
            box_idx = box_idx[still_alive]

        return boxes_to_keep

    def __get_nms_bboxes(self, score, y_c, x_c, height, width, iou_thresh, tiled_mode=False) -> np.array:
        """
        Performs the non-maximum suppression on the given bboxes

        :param score: the confidence score of the bbox (flatten array)
        :param y_c: the y coordinate of the center of the bbox (flatten array)
        :param x_c: the x coordinate of the center of the bbox (flatten array)
        :param height: the height of the bbox (flatten array)
        :param width: the width of the bbox (flatten array)
        :param iou_thresh: the minimum IoU threshold for the suppression
        :param tiled_mode:
        :return: an array of bboxes with the following structure: <score> <top> <left> <bottom> <right>
        """

        if tiled_mode:
            ymin, xmin, ymax, xmax = y_c, x_c, height, width
        else:
            # Flatten data
            score, coords, size = self.__flatten_bbox_data(score, y_c, x_c, height, width)
            xmin, ymin, xmax, ymax = coords

            # Take only boxes inside the image
            xmin, xmax, ymin, ymax, score = self.__take_boxes_inside_image(xmin, xmax, ymin, ymax, size, score)

        # Sort boxes by score in descending order
        score_sort = np.argsort(score)[::-1]
        score = score[score_sort]
        ymin = ymin[score_sort]
        xmin = xmin[score_sort]
        ymax = ymax[score_sort]
        xmax = xmax[score_sort]

        # Get all boxes
        boxes = np.concatenate((score.reshape(-1, 1),
                                ymin.reshape(-1, 1),
                                xmin.reshape(-1, 1),
                                ymax.reshape(-1, 1),
                                xmax.reshape(-1, 1)),
                               axis=1)

        # Return non-maximum-suppressed boxes
        return boxes[self.__nms(xmin, xmax, ymin, ymax, iou_thresh)]

    @staticmethod
    def __draw_rectangle(bbox_and_score: np.array, img: Image, color: str):
        """
        Utility to draw a bbox

        :param bbox_and_score: bidimensional array whose elements are in shape <ymin, xmin, ymax, xmax>
        :param img: the PIL image object in which to draw
        :param color: color of the box
        :return:
        """
        number_of_rect = np.minimum(500, len(bbox_and_score))

        for i in reversed(list(range(number_of_rect))):
            top, left, bottom, right = bbox_and_score[i, :]

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

            for j in range(2 * thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=rect_color)

        return img

    @staticmethod
    def __get_iou_score(true_boxes, predicted_boxes):
        """
        Compute mean IoU score for an image.
        """
        iou_all = []

        for predicted_box in predicted_boxes:
            y1 = np.maximum(predicted_box[0], true_boxes[:, 0])
            x1 = np.maximum(predicted_box[1], true_boxes[:, 1])
            y2 = np.minimum(predicted_box[2], true_boxes[:, 2])
            x2 = np.minimum(predicted_box[3], true_boxes[:, 3])

            # Intersection of true area with detected area
            intersection = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)

            # Sum of detected area + true area (intersection considered twice)
            sum_areas = (predicted_box[2] - predicted_box[0]) \
                        * (predicted_box[3] - predicted_box[1]) \
                        + (true_boxes[:, 2] - true_boxes[:, 0]) \
                        * (true_boxes[:, 3] - true_boxes[:, 1])

            # Union
            union = sum_areas - intersection

            # Compute IoU (single scalar)
            iou = np.amax(intersection / union)

            # Append the IoU to the list
            iou_all.append(iou)

        # Return mean score
        score = 2 * np.sum(iou_all) / (len(predicted_boxes) + len(true_boxes))

        return score
