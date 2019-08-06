from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path

import random
import pandas as pd
import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset:
    def __init__(self, path: str, image_size: int = 150, ratios: {} = None):
        """
        Initializes the dataset.
        :param path: the path to the dataset
        :param image_size: the size which the images in the dataset must be resized to
        :param ratios: the ratios of examples belonging to training, validation and test sets
        """

        self.path = Path(path)
        self.image_size = image_size

        # Set default values for ratios if no value has been provided
        self.ratios = ratios if ratios is not None else {
            'training': 0.6,
            'validation': 0.3
        }

        self.dataset, self.size = self.__build_dataset()

    def __build_dataset(self) -> (tf.data.Dataset, int):
        """
        Creates the actual dataset.
        :return: a dataset of images and its size
        """

        # Get all the paths to the images of the dataset
        all_image_paths = [os.path.join(self.path, image_name) for image_name in list(os.listdir(self.path))]
        dataset_size = len(all_image_paths)

        # Get all the labels of the images in the dataset
        pardir = os.path.join(self.path, os.pardir)
        all_image_labels = pd.read_csv(os.path.join(pardir, 'image_labels_map.csv'),
                                       header=0,
                                       usecols=['labels']).fillna('')
        all_image_labels = all_image_labels.values.tolist()

        # Create the dataset of all paths
        path_dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)

        # Create the dataset of all (preprocessed) images
        image_dataset = path_dataset.map(self.__load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

        # Create the dataset of all labels
        label_dataset = tf.data.Dataset.from_tensor_slices(all_image_labels)

        # Create the final dataset with both images and labels
        image_label_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

        return image_label_dataset, dataset_size

    def __load_and_preprocess_image(self, path):
        image = tf.read_file(path)
        return self.__preprocess_image(image)

    def __preprocess_image(self, image):
        # Decode the jpeg image
        image = tf.image.decode_jpeg(image, channels=1)

        # Resize the image
        image = tf.image.resize(image, [self.image_size, self.image_size])

        # Normalize to [0,1] range
        image /= 255.0

        return image

    def shuffle(self, buffer_size: int = 100, seed: int = int(random.random() * 100)):
        """
        Shuffles the dataset with the given seed.
        Note that multiple calls to this method may shuffle the dataset in the same way.
        :param buffer_size: the size of the buffer for the shuffling
        :param seed: the seed of the shuffling
        """

        self.dataset.shuffle(
            buffer_size=tf.cast(buffer_size, tf.int64),
            seed=tf.cast(seed, tf.int64),
            reshuffle_each_iteration=None
        )

    def get_size(self) -> int:
        return self.size

    def get_dataset(self) -> tf.data.Dataset:
        return self.dataset

    def get_training_set(self) -> tf.data.Dataset:
        # Take only the first training_ratio percent of the data
        return (self.dataset.take(tf.cast(self.size * self.ratios['training'], tf.int64))
                .batch(150)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_validation_set(self) -> tf.data.Dataset:
        # Take all but the first training_ratio percent of the data
        return (self.dataset
                .skip(tf.cast(self.size * self.ratios['training'], tf.int64))
                .take(tf.cast(self.size * self.ratios['validation'], tf.int64))
                .batch(150)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_test_set(self) -> tf.data.Dataset:
        # Take all but the first training_ratio + validation_ratio percent of the data
        return (self.dataset.skip(tf.cast(self.size * (self.ratios['training'] + self.ratios['validation']), tf.int64))
                .batch(150)
                .repeat()
                .prefetch(AUTOTUNE))

    def split(self) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
        return self.get_training_set(), self.get_validation_set(), self.get_test_set()
