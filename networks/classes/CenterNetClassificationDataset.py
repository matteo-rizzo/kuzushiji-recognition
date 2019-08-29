from typing import Dict, Generator, Tuple, List

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE


class ClassifierDataset:
    def __init__(self, params: Dict):
        self.__train_csv_path = params['train_csv_path']
        self.__train_images_path = params['train_images_path']
        self.__test_images_path = params['test_images_path']
        self.__sample_submission = params['sample_submission']
        self.__training_ratio = params['training_ratio']
        self.__batch_size = params['batch_size']
        self.__annotation_list_train: List[List]
        self.__aspect_ratio_pic_all: List[float]
        self.__input_height = params['input_height']
        self.__input_width = params['input_width']
        self.__output_height = params['output_height']
        self.__output_width = params['output_width']
        self.__validation_set: Tuple[tf.data.Dataset, int] = None
        self.__training_set: Tuple[tf.data.Dataset, int] = None

    def __dataset_generator(self, data_list: np.array, batch_size: int, is_train: bool = True,
                            random_crop: bool = True) \
            -> Generator:

        input_width, input_height = self.__input_width, self.__input_height

        x = []
        y = []

        count = 0

        while True:
            for sample in data_list:
                if random_crop:
                    crop_ratio = np.random.uniform(0.8, 1)
                else:
                    crop_ratio = 1
                with Image.open(sample[0]) as img:  # Image path

                    if random_crop and is_train:
                        img_width, img_height = img.size
                        img = np.asarray(img.convert('RGB'), dtype=np.uint8)
                        top_offset = np.random.randint(0, img_height - int(crop_ratio * img_height))
                        left_offset = np.random.randint(0, img_width - int(crop_ratio * img_width))
                        bottom_offset = top_offset + int(crop_ratio * img_height)
                        right_offset = left_offset + int(crop_ratio * img_width)
                        img = cv2.resize(img[top_offset:bottom_offset, left_offset, right_offset, :],
                                         (input_height, input_width))

                    else:
                        img = img.resize((input_width, input_height))
                        img = np.asarray(img.convert('RGB'), dtype=np.uint8)

                    x.append(img)

                    y.append(int(sample[1]))  # category

                count += 1

                if count == batch_size:
                    b_x = np.array(x, dtype=np.float32)
                    b_y = np.array(y, dtype=np.float32)  # ???????

                    b_x /= 255

                    count = 0
                    x = []
                    y = []

                    yield b_x, b_y

    def generate_dataset(self, input_list: List[Tuple[str, int]]) -> Tuple[List[List], List[List]]:
        """
        Generate the tf.data.Dataset containing all the objects.
        """

        X_train, X_test = train_test_split(input_list,
                                           train_size=int(self.__training_ratio * len(input_list)),
                                           shuffle=True)

        self.__training_set = (
            tf.data.Dataset.from_generator(
                lambda: self.__dataset_generator(X_train,
                                                 batch_size=self.__batch_size,
                                                 is_train=True,
                                                 random_crop=True),
                output_types=(np.float32,
                              np.float32))
                .repeat()
                .prefetch(AUTOTUNE),
            len(X_train))

        self.__validation_set = (
            tf.data.Dataset.from_generator(
                lambda: self.__dataset_generator(X_test,
                                                 batch_size=self.__batch_size,
                                                 is_train=False,
                                                 random_crop=False),
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

    def get_test_set(self, bbox_predictions: Dict[str, np.ndarray]):
        self.__test_set = (
            # Generate test set,
            len(bbox_predictions)
        )
