import tensorflow as tf

import numpy as np
from typing import List


class DataAugmentation:
    def __init__(self, augmentations: List[str]):
        augment = {
            'flip': self.__flip,
            'color': self.__color,
            'rotate': self.__rotate,
            'zoom': self.__zoom,
        }

        self.__augmentation_functions = [augment[augmentation] for augmentation in augmentations]

    def augment_dataset(self, dataset):
        for f in self.__augmentation_functions:
            dataset = dataset.map(f, num_parallel_calls=4)

        return dataset.map(lambda x: tf.clip_by_value(x, 0, 1))

    @staticmethod
    def __flip(x: tf.Tensor) -> tf.Tensor:
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        return x

    @staticmethod
    def __color(x: tf.Tensor) -> tf.Tensor:
        """
        Color augmentation

        :param x: Image
        :return Augmented image
        """
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return x

    @staticmethod
    def __rotate(x: tf.Tensor) -> tf.Tensor:
        """Rotation augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    @staticmethod
    def __zoom(x: tf.Tensor) -> tf.Tensor:
        """Zoom augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
            # Return a random crop
            return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

        choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

    @staticmethod
    def plot_images(dataset, n_images, samples_per_image):
        output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

        row = 0
        for images in dataset.repeat(samples_per_image).batch(n_images):
            output[:, row * 32:(row + 1) * 32] = np.vstack(images.numpy())
            row += 1

        plt.figure()
        plt.imshow(output)
        plt.show()
