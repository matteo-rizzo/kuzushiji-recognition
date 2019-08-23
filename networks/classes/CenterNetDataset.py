from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import io
import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CenterNetDataset:
    def __init__(self, params: Dict):
        self.__train_csv_path = params['train_csv_path']
        self.__train_images_path = params['train_images_path']
        self.__test_images_path = params['test_images_path']
        self.__sample_submission = params['sample_submission']
        self.__training_ratio = params['training_ratio']
        self.__batch_size = params['batch_size']
        self.__annotation_list_train: List[List]
        self.__aspect_ratio_pic_all: List[float]
        self.__dataset: Tuple[tf.data.Dataset, int]
        self.__input_height = params['input_height']
        self.__input_width = params['input_width']

    def __dataset_generator(self, filenames, batch_size):
        input_height, input_width = self.__input_height, self.__input_width
        category_n = 1
        output_layer_n = category_n + 4
        output_height, output_width = 128, 128

        x = []
        y = []

        count = 0

        while True:
            for i in range(len(filenames)):
                h_split = filenames[i][2]
                w_split = filenames[i][3]
                max_crop_ratio_h = 1 / h_split
                max_crop_ratio_w = 1 / w_split
                crop_ratio = np.random.uniform(0.5, 1)
                crop_ratio_h = max_crop_ratio_h * crop_ratio
                crop_ratio_w = max_crop_ratio_w * crop_ratio

                with Image.open(filenames[i][0]) as f:

                    # random crop

                    pic_width, pic_height = f.size
                    f = np.asarray(f.convert('RGB'), dtype=np.uint8)
                    top_offset = np.random.randint(0, pic_height - int(crop_ratio_h * pic_height))
                    left_offset = np.random.randint(0, pic_width - int(crop_ratio_w * pic_width))
                    bottom_offset = top_offset + int(crop_ratio_h * pic_height)
                    right_offset = left_offset + int(crop_ratio_w * pic_width)
                    f = cv2.resize(f[top_offset:bottom_offset, left_offset:right_offset, :],
                                   (input_height, input_width))
                    x.append(f)

                output_layer = np.zeros((output_height, output_width, (output_layer_n + category_n)))
                for annotation in filenames[i][1]:
                    x_c = (annotation[1] - left_offset) * (output_width / int(crop_ratio_w * pic_width))
                    y_c = (annotation[2] - top_offset) * (output_height / int(crop_ratio_h * pic_height))
                    width = annotation[3] * (output_width / int(crop_ratio_w * pic_width))
                    height = annotation[4] * (output_height / int(crop_ratio_h * pic_height))
                    top = np.maximum(0, y_c - height / 2)
                    left = np.maximum(0, x_c - width / 2)
                    bottom = np.minimum(output_height, y_c + height / 2)
                    right = np.minimum(output_width, x_c + width / 2)

                    if top >= (output_height - 0.1) or left >= (
                            output_width - 0.1) or bottom <= 0.1 or right <= 0.1:  # random crop(out of picture)
                        continue
                    width = right - left
                    height = bottom - top
                    x_c = (right + left) / 2
                    y_c = (top + bottom) / 2

                    category = 0  # not classify, just detect
                    heatmap = ((np.exp(
                        -(((np.arange(output_width) - x_c) / (width / 10)) ** 2) / 2)).reshape(1, -1)
                               * (np.exp(
                                -(((np.arange(output_height) - y_c) / (height / 10)) ** 2) / 2)).reshape(
                                -1, 1))
                    output_layer[:, :, category] = np.maximum(output_layer[:, :, category],
                                                              heatmap[:, :])
                    output_layer[int(y_c // 1), int(x_c // 1), category_n + category] = 1
                    output_layer[int(y_c // 1), int(x_c // 1), 2 * category_n] = y_c % 1  # height offset
                    output_layer[int(y_c // 1), int(x_c // 1), 2 * category_n + 1] = x_c % 1
                    output_layer[
                        int(y_c // 1), int(x_c // 1), 2 * category_n + 2] = height / output_height
                    output_layer[int(y_c // 1), int(x_c // 1), 2 * category_n + 3] = width / output_width
                y.append(output_layer)

                count += 1
                if count == batch_size:
                    x = np.array(x, dtype=np.float32)
                    y = np.array(y, dtype=np.float32)

                    inputs = x / 255
                    targets = y
                    x = []
                    y = []
                    count = 0
                    yield inputs, targets

    def generate_dataset(self, input_list):
        """
        Generate the tf.data.Dataset containing all the objects.
        """

        self.__dataset = (
            tf.data.Dataset.from_generator(self.__dataset_generator, (tf.float32, tf.float32),
                                           args=([input_list]))
                .shuffle(buffer_size=500), len(input_list))

    def get_training_set(self) -> Tuple[tf.data.Dataset, int]:
        TRAIN_SIZE = int(self.__training_ratio * self.__dataset[1])

        return (self.__dataset[0]
                .take(TRAIN_SIZE)
                .batch(self.__batch_size)
                .repeat()
                .prefetch(AUTOTUNE),
                TRAIN_SIZE)

    def get_validation_set(self) -> Tuple[tf.data.Dataset, int]:
        TRAIN_SIZE = int(self.__training_ratio * self.__dataset[1])

        return (self.__dataset[0]
                .skip(TRAIN_SIZE)
                .batch(self.__batch_size)
                .repeat()
                .prefetch(AUTOTUNE),
                self.__dataset[1] - TRAIN_SIZE)

    def get_test_set(self) -> Tuple[tf.data.Dataset, int]:
        # TODO
        return (None, 0)
