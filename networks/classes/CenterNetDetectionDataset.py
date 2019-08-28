from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CenterNetDataset:
    def __init__(self, params: Dict):
        self.__train_csv_path = params['train_csv_path']
        self.__train_images_path = params['train_images_path']
        self.__test_images_path = params['test_images_path']
        self.__sample_submission = params['sample_submission']

        self.__annotation_list_train: List[List]
        self.__aspect_ratio_pic_all: List[float]

        self.__training_ratio = params['training_ratio']
        self.__batch_size = params['batch_size']

        self.__input_height = params['input_height']
        self.__input_width = params['input_width']

        self.__output_height = params['output_height']
        self.__output_width = params['output_width']

        self.__validation_set: Tuple[tf.data.Dataset, int] = None
        self.__training_set: Tuple[tf.data.Dataset, int] = None

    def __dataset_generator(self, list_samples, batch_size) -> (np.float32, np.float32):
        """
        Generates a dataset given the samples and a batch size

        :param list_samples: the list of samples the dataset will contain
        :param batch_size:
        """

        input_height, input_width = self.__input_height, self.__input_width
        output_height, output_width = self.__output_height, self.__output_width

        category_n = 1
        output_layer_n = category_n + 4

        x = []
        y = []

        count = 0

        while True:
            for i in range(len(list_samples)):
                h_split = list_samples[i][2]  # recommended height split
                w_split = list_samples[i][3]
                max_crop_ratio_h = 1 / h_split
                max_crop_ratio_w = 1 / w_split
                crop_ratio = np.random.uniform(0.5, 1)
                crop_ratio_h = max_crop_ratio_h * crop_ratio
                crop_ratio_w = max_crop_ratio_w * crop_ratio

                with Image.open(list_samples[i][0]) as f:

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

                # list_samples[0] = path to image
                # list_samples[1] = ann
                # list_samples[2] = recommended height split
                # list_samples[3] = recommended width split
                # where 'ann' is data on bbox:
                # ann[:, 1] = xmin
                # ann[:, 2] = ymin
                # ann[:, 3] = x width
                # ann[:, 4] = y height

                for annotation in list_samples[i][1]:
                    x_c = (annotation[1] - left_offset) * (output_width / int(crop_ratio_w * pic_width))
                    y_c = (annotation[2] - top_offset) * (output_height / int(crop_ratio_h * pic_height))
                    width = annotation[3] * (output_width / int(crop_ratio_w * pic_width))
                    height = annotation[4] * (output_height / int(crop_ratio_h * pic_height))
                    top = np.maximum(0, y_c - height / 2)
                    left = np.maximum(0, x_c - width / 2)
                    bottom = np.minimum(output_height, y_c + height / 2)
                    right = np.minimum(output_width, x_c + width / 2)

                    if top >= (output_height - 0.1) or left >= (output_width - 0.1) \
                            or bottom <= 0.1 or right <= 0.1:  # random crop (out of picture)
                        continue
                    width = right - left
                    height = bottom - top
                    x_c = (right + left) / 2
                    y_c = (top + bottom) / 2

                    # Gaussian kernel
                    heatmap = (
                            (np.exp(
                                -(((np.arange(output_width) - x_c) / (width / 10)) ** 2) / 2
                            ))
                            .reshape(1, -1) *
                            (np.exp(
                                -(((np.arange(output_height) - y_c) / (height / 10)) ** 2) / 2
                            ))
                            .reshape(-1, 1)
                    )
                    # Center points will have value closer to 1 (close to 0 otherwise)

                    # In original paper heatmap is computed:
                    # exp(-((x_distance_from_center) ^ 2 + (y_distance_from_center) ^ 2) / (2 * sig ^ 2))
                    # Here x_distance_from_center = np.arange(output_width) - x_c
                    # and y_distance_from_center = np.arange(output_height) - y_c
                    # sigma = (width / 10) and (height / 10).
                    # Multiplying allows to consider the shape of characters as ellipses, instead of
                    # circles. Result is 2D array

                    # category heatmap
                    output_layer[:, :, 0] = np.maximum(output_layer[:, :, 0], heatmap[:, :])
                    output_layer[int(y_c // 1), int(x_c // 1), 1] = 1
                    output_layer[int(y_c // 1), int(x_c // 1), 2] = y_c % 1  # height offset
                    output_layer[int(y_c // 1), int(x_c // 1), 3] = x_c % 1
                    output_layer[int(y_c // 1), int(x_c // 1), 4] = height / output_height
                    output_layer[int(y_c // 1), int(x_c // 1), 5] = width / output_width

                y.append(output_layer)

                # print(output_layer.shape)

                count += 1
                if count == batch_size:
                    x = np.array(x, dtype=np.float32)
                    y = np.array(y, dtype=np.float32)

                    # print('Yield batch shape ', y.shape)

                    inputs = x / 255
                    targets = y
                    x = []
                    y = []
                    count = 0

                    yield inputs, targets

    def generate_dataset(self, input_list) -> Tuple[List[List], List[List]]:
        """
        Generate the tf.data.Dataset containing all the objects.
        """

        X_train, X_test = train_test_split(input_list,
                                           train_size=int(self.__training_ratio * len(input_list)))

        self.__training_set = (
            tf.data.Dataset.from_generator(
                lambda: self.__dataset_generator(X_train, self.__batch_size),
                output_types=(np.float32,
                              np.float32))
                .repeat()
                .prefetch(AUTOTUNE),
            len(X_train))

        self.__validation_set = (
            tf.data.Dataset.from_generator(
                lambda: self.__dataset_generator(X_test, self.__batch_size),
                output_types=(np.float32,
                              np.float32))
                .repeat()
                .prefetch(AUTOTUNE),
            len(X_test))

        return X_train, X_test

    def get_training_set(self) -> Tuple[tf.data.Dataset, int]:
        return self.__training_set

    def get_validation_set(self) -> Tuple[tf.data.Dataset, int]:
        return self.__validation_set

    def get_test_set(self) -> Tuple[tf.data.Dataset, int]:
        # TODO
        return (None, 0)
