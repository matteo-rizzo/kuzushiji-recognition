from math import exp
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


class PreprocessingDataset:
    def __init__(self, params: Dict):
        self.__train_csv_path = params['train_csv_path']
        self.__train_images_path = params['train_images_path']

        self.__input_height = params['input_height']
        self.__input_width = params['input_width']
        self.__training_ratio = params['training_ratio']
        self.__batch_size = params['batch_size']

        self.__train_list: List[Tuple[str, np.array]] = []
        self.__w_and_h: List[float] = []
        self.__dict_cat: Dict[str, int] = {}

        self.__dataset: Tuple[tf.data.Dataset, int]

    def get_categories_dict(self) -> Dict[str, int]:
        return self.__dict_cat

    def generate_dataset(self):
        # Generate a train list of tuples where each row represents an image and the list of the
        # characters within it (codified as integers) with relative coordinates of bbox
        self.__parse_train_csv()

        # Compute the average bbox ratio w.r.t. the image area considering all the images
        self.__annotate_char_area_ratio(show_plot=False)

        # Generate the tf.data.Dataset containing all the objects
        self.__compose_dataset_object()

    def __get_dataset_labels(self) -> List[float]:
        return [img_data[1] for img_data in self.__train_image_avg_char_area_ratios]

    def __parse_train_csv(self):
        """
        Read training csv and generate a list with char classes and bounding box in a convenient format.
        """
        df_train = pd.read_csv(self.__train_csv_path)

        # Remove any row with at least one nan value
        df_train = df_train.dropna(axis=0, how='any')
        df_train = df_train.reset_index(drop=True)

        category_names = set()

        for i in range(len(df_train)):
            # Get one row for each label character for image i,
            # as: <category> <x> <y> <width> <height>
            ann = np.array(df_train.loc[i, "labels"].split(" ")).reshape(-1, 5)

            # Get set of unique categories
            category_names = category_names.union({i for i in ann[:, 0]})

        category_names = sorted(category_names)

        # Make a dict assigning an integer to each category
        self.__dict_cat: Dict[str, int] = {list(category_names)[j]: j for j in range(len(category_names))}

        for i in range(len(df_train)):
            # Get one row for each label character for image i, as: <category> <x> <y> <width> <height>
            ann = np.array(df_train.loc[i, "labels"].split(" ")).reshape(-1, 5)

            # Iterate over categories in first column of annotations (characters)
            for j, category_name in enumerate(ann[:, 0]):
                # Change categories in integer values
                ann[j, 0] = int(self.__dict_cat[category_name])

            # Calculate the center of each bbox
            # Before the operations:
            # - ann[:, 0] = class
            # - ann[:, 1] = xmin
            # - ann[:, 2] = ymin
            # - ann[:, 3] = x width
            # - ann[:, 4] = y height

            ann = ann.astype('int32')

            # center_x
            ann[:, 1] += ann[:, 3] // 2

            # center_y
            ann[:, 2] += ann[:, 4] // 2

            # After the operations:
            #    ann[:, 0] = class
            # -> ann[:, 1] = x_center
            # -> ann[:, 2] = y_center
            #    ann[:, 3] = x width
            #    ann[:, 4] = y height

            self.__train_list.append(("{}/{}.jpg".format(self.__train_images_path, df_train.loc[i, "image_id"]),
                                      ann))

    def __annotate_char_area_ratio(self, show_plot: bool = False):
        """
        Computes the average bbox ratio w.r.t. the image img_area considering all the training images,
        and optionally plots a graph with ratio distribution.

        :param show_plot: whether to show histogram with average character size or not.
        """

        self.__w_and_h: List[(float, float)] = []
        self.__train_image_avg_char_area_ratios: List[Tuple[str, np.array]] = []
        all_avg_char_area_ratio: List[np.array] = []

        for img_path, ann in self.__train_list:
            with Image.open(img_path) as img:
                # Image dimensions
                width, height = img.size

                # Image img_area
                img_area = width * height
                # aspect_ratio = height / width

                # Bbox img_area for each character in image (width * height)
                char_area = ann[:, 3] * ann[:, 4]

                # List of ratios for each character
                char_area_ratio = char_area / img_area

                # Take the mean and append it to a list
                avg_char_area_ratio = np.mean(char_area_ratio)
                all_avg_char_area_ratio.append(avg_char_area_ratio)

                # Add example for training with image path and log average bbox size for objects in it
                self.__train_image_avg_char_area_ratios.append((img_path, np.log(avg_char_area_ratio)))

                # Add aspect ratio
                self.__w_and_h.append((width, height))

        if show_plot:
            plt.hist(np.log(all_avg_char_area_ratio), bins=100)
            plt.title('log(ratio of char_area / picture_size)', loc='center', fontsize=12)
            plt.show()

    def get_crop_values(self) -> List[Tuple[str, np.array, float, float]]:
        """
        Given a list of sizes of bboxes for each train image,
        computes the best size according to the image must be split

        :return: extended annotation list with recommended splits in format:
            [image path, annotations, height split, width split]
        """

        # Initialize a list of bbox areas for all characters (prediction or from train data)
        avg_log_char_area_ratios: List[float] = self.__get_dataset_labels()

        # Set a base stretch factor
        base_stretch_factor_h, base_stretch_factor_w = 25, 25

        # Initialize an empty list of annotations
        annotation_list_train_w_crops = []

        # For each predicted bbox size calculate recommended height and width
        for img_ann, avg_char_area_ratio, img_wh in zip(self.__train_list, avg_log_char_area_ratios, self.__w_and_h):
            # Get width and height of the image
            w, h = img_wh

            # stretch_factor_h = (h / w) * sqrt(h*w / mean_char_h*mean_char_w)
            stretch_factor_h = (h / w) * np.sqrt(h * w / avg_char_area_ratio)

            # stretch_factor_w = sqrt(h*w / mean_char_h*mean_char_w)
            stretch_factor_w = np.sqrt(h * w / avg_char_area_ratio)

            # If image is too big w.r.t. mean char size then stretch it with factor > 1
            h_split_recommend = max([1, stretch_factor_h / base_stretch_factor_h])
            w_split_recommend = max([1, stretch_factor_w / base_stretch_factor_w])

            # Format: <image path> <annotations> <height split> <width split>
            annotation_list_train_w_crops.append((img_ann[0],
                                                  img_ann[1],
                                                  h_split_recommend,
                                                  w_split_recommend))

        return annotation_list_train_w_crops

    def __preprocess_image(self, image, label, is_train=True, random_crop=True):
        """
        Processes an image

        :param image: tensor representing image
        :param label: tensor representing label
        :param is_train: true if preprocessing for training set, else false
        :param random_crop: whether to apply a random crop
        :return: image and labels tensors.
        """

        input_width, input_height = self.__input_width, self.__input_height
        crop_ratio = np.random.uniform(0.7, 1) if random_crop else 1

        # Load image
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_jpeg(image_string)

        if random_crop and is_train:

            # Get image size
            pic_height, pic_width, _ = image_decoded.get_shape().as_list()

            # Compute the offsets
            top_offset = np.random.randint(0, pic_height - int(crop_ratio * pic_height)) / (pic_height - 1)
            left_offset = np.random.randint(0, pic_width - int(crop_ratio * pic_width)) / (pic_width - 1)
            bottom_offset = top_offset + int(crop_ratio * pic_height) / (pic_height - 1)
            right_offset = left_offset + int(crop_ratio * pic_width) / (pic_width - 1)

            # Resize the image
            image_resized = tf.image.crop_and_resize(image=[image_decoded],
                                                     box_ind=[0],
                                                     boxes=[[top_offset,
                                                             left_offset,
                                                             bottom_offset,
                                                             right_offset]],
                                                     crop_size=[input_width, input_height])

            # Update average bbox size after cropping
            label -= np.log(crop_ratio)
        else:
            image_resized = tf.image.resize_images(images=image_decoded,
                                                   size=[input_width, input_height])

        # Make sure values are in range [0, 255]
        image_resized /= 255

        # Remove 1st dimension if image has been cropped: (1, height, width, 3) -> (height, width, 3)
        if random_crop:
            image_resized = tf.reshape(image_resized, image_resized.shape[1:])

        return image_resized, label

    def __compose_dataset_object(self):
        """
        Generates the tf.data.Dataset containing all the objects
        """

        # Iterate over paths of images
        image_paths = [sample[0] for sample in self.__train_image_avg_char_area_ratios]

        # Iterate over avg bbox ratios
        image_labels = [sample[1] for sample in self.__train_image_avg_char_area_ratios]

        image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        label_dataset = tf.data.Dataset.from_tensor_slices(image_labels)

        self.__dataset = (tf.data.Dataset.zip((image_dataset, label_dataset)).shuffle(buffer_size=500),
                          len(image_paths))

    def get_training_set(self) -> Tuple[tf.data.Dataset, int]:
        train_size = int(self.__training_ratio * self.__dataset[1])

        return (self.__dataset[0]
                .take(train_size)
                .map(lambda path, label: tf.py_function(self.__preprocess_image,
                                                        [path, label, True, True],
                                                        (tf.float32, tf.float64)),
                     num_parallel_calls=AUTOTUNE)
                .batch(self.__batch_size)
                .repeat()
                .prefetch(AUTOTUNE),
                train_size)

    def get_validation_set(self) -> Tuple[tf.data.Dataset, int]:
        train_size = int(self.__training_ratio * self.__dataset[1])

        return (self.__dataset[0]
                .skip(train_size)
                .map(lambda path, label: tf.py_function(self.__preprocess_image,
                                                        [path, label, False, False],
                                                        (tf.float32, tf.float64)),
                     num_parallel_calls=AUTOTUNE)
                .batch(self.__batch_size)
                .repeat()
                .prefetch(AUTOTUNE),
                self.__dataset[1] - train_size)
