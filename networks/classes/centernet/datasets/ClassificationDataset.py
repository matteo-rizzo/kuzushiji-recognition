from typing import Dict, Generator, Tuple, List, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE


class ClassificationDataset:
    def __init__(self, params: Dict):
        self.__train_csv_path = params['train_csv_path']
        self.__train_images_path = params['train_images_path']
        self.__test_images_path = params['test_images_path']
        self.__sample_submission = params['sample_submission']

        self.__training_ratio = params['training_ratio']
        self.__validation_ratio = params['validation_ratio']
        self.__evaluation_ratio = params['evaluation_ratio']
        self.__batch_size = params['batch_size']
        self.__batch_size_predict = params['batch_size_predict']

        self.__annotation_list_train: List[List]
        self.__aspect_ratio_pic_all: List[float]

        self.__input_height = params['input_height']
        self.__input_width = params['input_width']
        self.__output_height = params['output_height']
        self.__output_width = params['output_width']

        self.__validation_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)
        self.__evaluation_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)
        self.__training_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)
        self.__test_set: Tuple[Union[tf.data.Dataset, None], int] = (None, 0)

    def __dataset_generator(self,
                            data_list: np.array,
                            is_train: bool = True,
                            random_crop: bool = True) -> Generator:

        input_width, input_height = self.__input_width, self.__input_height

        x = []
        y = []

        count = 0

        while True:
            for sample in data_list:

                crop_ratio = np.random.uniform(0.8, 1) if random_crop else 1

                with Image.open(sample[0]) as img:

                    if random_crop and is_train:
                        img_width, img_height = img.size
                        img = np.asarray(img.convert('RGB'), dtype=np.uint8)

                        top_offset = np.random.randint(0, img_height - int(crop_ratio * img_height))
                        left_offset = np.random.randint(0, img_width - int(crop_ratio * img_width))
                        bottom_offset = top_offset + int(crop_ratio * img_height)
                        right_offset = left_offset + int(crop_ratio * img_width)

                        img = cv2.resize(img[top_offset:bottom_offset, left_offset:right_offset, :],
                                         (input_height, input_width))

                    else:
                        img = img.resize((input_width, input_height))
                        img = np.asarray(img.convert('RGB'), dtype=np.uint8)

                    # Append the current image
                    x.append(img)

                    # Append the category of the current image
                    y.append(int(sample[1]))

                count += 1

                if count == self.__batch_size:
                    b_x = np.array(x, dtype=np.float32)
                    b_y = np.array(y, dtype=np.float32)

                    b_x /= 255

                    count = 0

                    x = []
                    y = []

                    yield b_x, b_y

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
                         train_list: List[Tuple[str, int]],
                         test_list: Union[List[str], None]) -> Tuple[List[List], List[List], List[List]]:

        """
        Generate the tf.data.Dataset containing all the objects.

        :param train_list: training list with samples as list of tuples (image, class)
        :param test_list: a list of test image paths, ore None if test set must not be generated.
        :return: the split and shuffled train and validation set, in the same shape as 'train_list'
                param.
        """

        assert self.__evaluation_ratio + self.__training_ratio + self.__validation_ratio == 1, \
            'Split ratios are not correctly set up'

        training, xy_eval = train_test_split(train_list,
                                             random_state=797,
                                             shuffle=True,
                                             train_size=int(
                                                 (1 - self.__evaluation_ratio) * len(train_list)))

        xy_train, xy_val = train_test_split(training,
                                            train_size=int(self.__training_ratio * len(train_list)),
                                            shuffle=True)

        self.__training_set = (
            tf.data.Dataset.from_generator(
                lambda: self.__dataset_generator(xy_train,
                                                 is_train=True,
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
                                                     is_train=False,
                                                     random_crop=False),
                    output_types=(np.float32,
                                  np.float32))
                    .repeat()
                    .prefetch(AUTOTUNE),
                len(xy_val))

        if len(xy_eval):
            self.__evaluation_set = (
                tf.data.Dataset.from_generator(
                    lambda: self.__dataset_generator(xy_eval,
                                                     is_train=False,
                                                     random_crop=False),
                    output_types=(np.float32,
                                  np.float32))
                    .repeat()
                    .prefetch(AUTOTUNE),
                len(xy_val))

        if test_list is not None:
            self.__test_set = (
                tf.data.Dataset.from_tensor_slices(test_list)
                    .map(self.__test_resize_fn,
                         num_parallel_calls=AUTOTUNE)
                    .batch(self.__batch_size_predict)
                    .prefetch(AUTOTUNE),
                len(test_list)
            )
        # else: it remains (None, 0)

        return xy_train, xy_val, xy_eval

    def get_training_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__training_set

    def get_validation_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__validation_set

    def get_evaluation_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__evaluation_set

    def get_test_set(self) -> Tuple[Union[tf.data.Dataset, None], int]:
        return self.__test_set
