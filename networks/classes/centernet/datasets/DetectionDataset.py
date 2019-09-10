from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DetectionDataset:
    def __init__(self, params: Dict):
        self.__annotation_list_train: List[List]
        self.__aspect_ratio_pic_all: List[float]

        self.__training_ratio = params['training_ratio']
        self.__validation_ratio = params['validation_ratio']
        self.__evaluation_ratio = params['evaluation_ratio']
        self.__batch_size = params['batch_size']
        self.__batch_size_predict = params['batch_size_predict']

        self.__input_height = params['input_height']
        self.__input_width = params['input_width']

        self.__output_height = params['output_height']
        self.__output_width = params['output_width']

        self.__validation_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)
        self.__training_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)
        self.__evaluation_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)
        self.__test_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)

    def __dataset_generator(self,
                            list_samples,
                            batch_size,
                            random_crop: bool = True) -> (np.float32, np.float32):
        """
        Generates a dataset given the samples and a batch size

        :param list_samples: the list of samples the dataset will contain
        :param batch_size:

        Note that in the original paper, the heatmap is computed as:
        * exp(-((x_distance_from_center) ^ 2 + (y_distance_from_center) ^ 2) / (2 * sig ^ 2))

        Here:
        - x_distance_from_center = np.arange(output_width) - x_c
        - y_distance_from_center = np.arange(output_height) - y_c
        - sigma = (width / 10) and (height / 10).

        Multiplying allows to consider the shape of characters as ellipses, instead of circles.
        The result is a 2D array
        """

        input_height, input_width = self.__input_height, self.__input_width
        output_height, output_width = self.__output_height, self.__output_width

        category_n = 1
        output_layer_n = category_n + 4

        x, y = [], []

        count = 0

        while True:
            for i in range(len(list_samples)):
                h_split = list_samples[i][2]
                w_split = list_samples[i][3]

                max_crop_ratio_h = 1 / h_split
                max_crop_ratio_w = 1 / w_split

                crop_ratio = np.random.uniform(0.5, 1)
                crop_ratio_h = max_crop_ratio_h * crop_ratio
                crop_ratio_w = max_crop_ratio_w * crop_ratio

                with Image.open(list_samples[i][0]) as f:

                    pic_width, pic_height = f.size

                    if random_crop:
                        f = np.asarray(f.convert('RGB'), dtype=np.uint8)

                        top_offset = np.random.randint(0, pic_height - int(crop_ratio_h * pic_height))
                        left_offset = np.random.randint(0, pic_width - int(crop_ratio_w * pic_width))
                        bottom_offset = top_offset + int(crop_ratio_h * pic_height)
                        right_offset = left_offset + int(crop_ratio_w * pic_width)

                        f = cv2.resize(f[top_offset:bottom_offset, left_offset:right_offset, :],
                                       (input_height, input_width))
                    else:
                        # No crop
                        top_offset, left_offset, bottom_offset, right_offset = 0, 0, 0, 0
                        crop_ratio, crop_ratio_h, crop_ratio_w = 1, 1, 1
                        f = f.resize((input_width, input_height))
                        f = np.asarray(f.convert('RGB'), dtype=np.uint8)

                    x.append(f)

                output_layer = np.zeros((output_height, output_width, (output_layer_n + category_n)))

                # list_samples has the following structure:
                # list_samples[0] = path to image
                # list_samples[1] = ann
                # list_samples[2] = recommended height split
                # list_samples[3] = recommended width split
                # Where 'ann' the bbox data:
                # ann[:, 1] = xmin
                # ann[:, 2] = ymin
                # ann[:, 3] = x width
                # ann[:, 4] = y height

                for annotation in list_samples[i][1]:
                    x_c = (annotation[1] - left_offset) * (output_width / int(crop_ratio_w * pic_width))
                    y_c = (annotation[2] - top_offset) * (output_height / int(crop_ratio_h * pic_height))

                    # Divide by output stride (pic_width * crop / out_width)
                    width = annotation[3] * (output_width / int(crop_ratio_w * pic_width))
                    height = annotation[4] * (output_height / int(crop_ratio_h * pic_height))

                    top = np.maximum(0, y_c - height / 2)
                    left = np.maximum(0, x_c - width / 2)
                    bottom = np.minimum(output_height, y_c + height / 2)
                    right = np.minimum(output_width, x_c + width / 2)

                    # Random crop (out of picture)
                    if top >= (output_height - 0.1) or left >= (output_width - 0.1) \
                            or bottom <= 0.1 or right <= 0.1:
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

                    # Category heatmap
                    output_layer[:, :, 0] = np.maximum(output_layer[:, :, 0], heatmap[:, :])
                    output_layer[int(y_c // 1), int(x_c // 1), 1] = 1
                    output_layer[int(y_c // 1), int(x_c // 1), 2] = y_c % 1  # height offset
                    output_layer[int(y_c // 1), int(x_c // 1), 3] = x_c % 1
                    output_layer[int(y_c // 1), int(x_c // 1), 4] = height / output_height
                    output_layer[int(y_c // 1), int(x_c // 1), 5] = width / output_width

                y.append(output_layer)

                count += 1
                if count == batch_size:
                    x = np.array(x, dtype=np.float32)
                    y = np.array(y, dtype=np.float32)

                    inputs = x / 255
                    targets = y

                    x, y = [], []

                    count = 0

                    yield inputs, targets

    def __test_resize_fn(self, path):
        """
        Utility function for image resizing

        :param path: the path to the image to be resized
        :return: a resized image
        """

        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (self.__input_height, self.__input_width))

        return image_resized / 255

    def generate_dataset(self,
                         train_list: List[List],
                         test_list: Union[List[str], None]) -> Tuple[List[List], List[List], List[List]]:
        """
        Generate the tf.data.Dataset with train, validation and test set.

        :param train_list: list with format [[image path, annotations, height split, width split]]
        :param test_list: list of test images file paths, or None if we don't want to generate the
            test set (i.e. not in predict mode).
        :return: the train and validation sets in the same format as the input_list, after splitting
            and shuffling operations.
        """

        assert self.__evaluation_ratio + self.__training_ratio + self.__validation_ratio == 1, \
            'Split ratios are not correctly set up!'

        training, xy_eval = train_test_split(train_list,
                                             random_state=797,
                                             shuffle=True,
                                             train_size=int((1 - self.__evaluation_ratio) * len(train_list)))

        xy_train, xy_val = train_test_split(training,
                                            random_state=55373,
                                            shuffle=True,
                                            train_size=int(self.__training_ratio * len(train_list)))

        self.__training_set = (
            tf.data.Dataset.from_generator(
                lambda: self.__dataset_generator(xy_train,
                                                 self.__batch_size,
                                                 random_crop=True),
                output_types=(np.float32,
                              np.float32))
                .repeat()
                .prefetch(AUTOTUNE),
            len(xy_train))

        if len(xy_val):
            self.__validation_set = (
                tf.data.Dataset.from_generator(
                    lambda: self.__dataset_generator(xy_val,
                                                     self.__batch_size,
                                                     random_crop=True),
                    output_types=(np.float32,
                                  np.float32))
                    .repeat()
                    .prefetch(AUTOTUNE),
                len(xy_val))

        if len(xy_eval):
            self.__evaluation_set = (
                tf.data.Dataset.from_generator(
                    lambda: self.__dataset_generator(xy_eval,
                                                     self.__batch_size,
                                                     random_crop=False),
                    output_types=(np.float32,
                                  np.float32))
                    .repeat()
                    .prefetch(AUTOTUNE),
                len(xy_eval))

        if test_list is not None:
            self.__test_set = (
                tf.data.Dataset.from_tensor_slices(test_list)
                    .map(self.__test_resize_fn, num_parallel_calls=AUTOTUNE)
                    .batch(self.__batch_size_predict)
                    .prefetch(AUTOTUNE),
                len(test_list))
            # else: it's (None, 0)

        return xy_train, xy_val, xy_eval

    def get_training_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__training_set

    def get_validation_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__validation_set

    def get_evaluation_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__evaluation_set

    def get_test_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__test_set
