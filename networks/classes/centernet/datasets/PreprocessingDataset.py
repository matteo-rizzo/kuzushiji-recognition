from typing import Dict, List, Tuple
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from math import log

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
        self.__class_weights: Dict[int, float] = {}

        self.__dataset: Tuple[tf.data.Dataset, int]

    def get_categories_dict(self) -> Dict[str, int]:
        return self.__dict_cat

    def get_class_weights(self) -> Dict[int, float]:
        return self.__class_weights

    def generate_dataset(self):
        # Generate a train list of tuples where each row represents an image and the list of the
        # characters within it (codified as integers) with relative coordinates of bbox
        self.__parse_train_csv()

        # Compute the average bbox ratio w.r.t. the image area considering all the images
        self.__annotate_char_area_ratio()

        # Generate the tf.data.Dataset containing all the objects
        self.__compose_dataset_object()

    def __get_dataset_labels(self) -> List[float]:
        return [img_data[1] for img_data in self.__train_image_avg_char_area_ratios]

    def __set_class_encoding(self, df_train, show_frequency=False):
        # Initialize an empty dictionary of <unicode> -> <class frequency>
        class_frequencies = {}

        # Iterate over the rows of the train list
        for i in range(len(df_train)):
            # Get one row for each label character w.r.t. image i, as: <category> <x> <y> <width> <height>
            ann = np.array(df_train.loc[i, "labels"].split(" ")).reshape(-1, 5)

            # Iterate over the unicodes of the current row
            for unicode in ann[:, 0]:
                # Update the frequency of class <unicode>
                frequency = class_frequencies.setdefault(unicode, 0) + 1
                class_frequencies[unicode] = frequency

        # Sort the dictionary of frequencies
        ord_class_frequencies = OrderedDict(sorted(class_frequencies.items()))

        # Make a dict assigning an integer to each category
        category_names = ord_class_frequencies.keys()
        self.__dict_cat: Dict[str, int] = {list(category_names)[j]: j for j in range(len(category_names))}

        # Set up a dictionary of frequencies as: <int class code> -> <frequency>
        dict_frequencies = {self.__dict_cat[k]: ord_class_frequencies[k] for k in self.__dict_cat.keys()}

        # Set up the dictionary of class weights as: <int class code> -> <relative frequency>
        num_occurs = sum(dict_frequencies.values())
        max_frequency = max(dict_frequencies.values()) * 100 / num_occurs
        eps = 0.00005
        self.__class_weights = {k: log(max_frequency / (v * 100 / num_occurs) + eps) for k, v in dict_frequencies.items()}

        if show_frequency:
            cat, frequency = zip(*dict_frequencies.items())
            plt.plot(cat, frequency, color='red')
            plt.title('Class frequencies', fontsize=18)
            plt.xlabel('Class (mapped to integer)', fontsize=14)
            plt.ylabel('Absolute class frequency', fontsize=14)
            plt.show()
            plt.clf()

            cat, rel_frequency = zip(*self.__class_weights.items())
            plt.bar(cat, rel_frequency, color='blue')
            plt.title('Log relative class frequencies', fontsize=16)
            plt.xlabel('Class', fontsize=14)
            plt.ylabel('Log relative frequency', fontsize=14)
            plt.show()

    def __parse_train_csv(self):
        """
        Read training csv and generate a list with char classes and bounding box in a convenient format.
        """

        # Read the train list from csv
        df_train = pd.read_csv(self.__train_csv_path)

        # Remove any row with at least one nan value
        df_train = df_train.dropna(axis=0, how='any')
        df_train = df_train.reset_index(drop=True)

        self.__set_class_encoding(df_train)

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

            self.__train_list.append(
                ("{}/{}.jpg".format(self.__train_images_path, df_train.loc[i, "image_id"]),
                 ann))

    def __annotate_char_area_ratio(self, show_plot: bool = False):
        """
        Computes the average bbox ratio w.r.t. the image img_area considering all the training images,
        and optionally plots a graph with ratio distribution.

        :param show_plot: whether to show histogram with average character size or not.
        """

        self.__aspect_ratios: List[float] = []
        self.__train_image_avg_char_area_ratios: List[Tuple[str, np.array]] = []
        all_avg_char_area_ratio: List[np.array] = []

        for img_path, ann in self.__train_list:
            with Image.open(img_path) as img:
                # Image dimensions
                width, height = img.size

                # Image img_area
                img_area = width * height
                aspect_ratio = height / width

                # Bbox img_area for each character in image (width * height)
                char_area = ann[:, 3] * ann[:, 4]

                # List of ratios for each character
                char_area_ratio = char_area / img_area

                # Take the mean and append it to a list
                avg_char_area_ratio = np.mean(char_area_ratio)
                all_avg_char_area_ratio.append(avg_char_area_ratio)

                # Add example for training with image path and log average bbox size for objects in it
                self.__train_image_avg_char_area_ratios.append((img_path, avg_char_area_ratio))

                # Add aspect ratio
                self.__aspect_ratios.append(aspect_ratio)

        if show_plot:
            plt.hist(np.log(all_avg_char_area_ratio), bins=100)
            plt.title('log(ratio of char_area / picture_size)', loc='center', fontsize=12)
            plt.show()

    def get_crop_values(self) -> List[Tuple[str, np.array, float, float]]:
        """
        Given a list of sizes of bboxes for each train image, computes the best size the image must be split to

        :return: extended annotation list with recommended splits in format:
            [image path, annotations, height split, width split]
        """

        # Initialize a list of bbox areas for all characters (prediction or from train data)
        avg_char_area_ratios: List[float] = self.__get_dataset_labels()

        # Set a base stretch factor
        base_split_factor_h, base_split_factor_w = 25, 25

        # Initialize an empty list of annotations
        annotation_list_train_w_crops = []

        # For each predicted bbox size calculate recommended height and width
        for img_ann, avg_char_area_ratio, aspect_ratio in zip(self.__train_list,
                                                              avg_char_area_ratios,
                                                              self.__aspect_ratios):
            # split_factor_w = sqrt(h*w / mean_char_h*mean_char_w)
            split_factor_w = np.sqrt(1.0 / avg_char_area_ratio)

            # split_factor_h = (h / w) * sqrt(h*w / mean_char_h*mean_char_w)
            split_factor_h = aspect_ratio * split_factor_w

            # If image is too big w.r.t. mean char size then stretch it with factor > 1
            h_crop_ratio = max([1, split_factor_h / base_split_factor_h])
            w_crop_ratio = max([1, split_factor_w / base_split_factor_w])

            # Format: <image path> <annotations> <height split> <width split>
            annotation_list_train_w_crops.append((img_ann[0],
                                                  img_ann[1],
                                                  h_crop_ratio,
                                                  w_crop_ratio))

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
            top_offset = np.random.randint(0, pic_height - int(crop_ratio * pic_height)) / (
                    pic_height - 1)
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
