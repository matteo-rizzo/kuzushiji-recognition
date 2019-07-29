from __future__ import absolute_import, division, print_function, unicode_literals
import random
from pathlib import Path
import pandas as pd
import os

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, path: str):
        """
        Initialize the dataset.
        :param path: the path to the dataset
        """
        self.path = Path(path)
        self.dataset = self.__build_dataset()

    def __build_dataset(self) -> tf.data.Dataset:
        # Get all the paths to the images of the dataset
        all_image_paths = list(os.listdir(self.path))
        all_image_paths = [str(path) for path in all_image_paths]

        # Get all the labels of the images in the dataset
        all_image_labels = pd.read_csv('datasets/kaggle/train.csv', usecols=['labels'])

        # TODO: remove
        print(len(all_image_paths))
        print(all_image_paths[:10])
        print(all_image_labels.head())

        path_dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_dataset = path_dataset.map(self.__load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        label_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
        image_label_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

        # TODO: remove
        print(image_label_dataset)

        return image_label_dataset

    def __load_and_preprocess_image(self, path):

        image = tf.read_file(path)
        return self.__preprocess_image(image)

    @staticmethod
    def __preprocess_image(image):
        # Decode the jpeg image
        image = tf.image.decode_jpeg(image, channels=3)

        # Resize all the image
        image = tf.image.resize(image, [192, 192])

        # Normalize to [0,1] range
        image /= 255.0

        return image

    def shuffle(self, seed: int = int(random.random() * 100)):
        """
        Shuffles the dataset with the given seed.
        Note that multiple calls to this method may shuffle the dataset in the same way.
        :param seed: the seed of the shuffling
        """
        random.Random(seed).shuffle(self.dataset)

    def get_training_set(self, ratio=0.6) -> tf.data.Dataset:
        pass

    def get_validation_set(self, ratio=0.2) -> tf.data.Dataset:
        pass

    def get_test_set(self, ratio=0.1) -> tf.data.Dataset:
        pass

    def split(self, test_ratio, validation_ratio) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        """
        Split the dataset into training, validation and test sets
        :return: training, validation and test sets
        """
        return self.get_training_set(test_ratio), \
               self.get_validation_set(validation_ratio), \
               self.get_test_set(abs(test_ratio - validation_ratio))
