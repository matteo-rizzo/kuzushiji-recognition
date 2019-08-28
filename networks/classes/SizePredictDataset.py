from math import exp
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


class SizePredictDataset:
    def __init__(self, params: Dict):
        self.__train_csv_path = params['train_csv_path']
        self.__train_images_path = params['train_images_path']
        self.__test_images_path = params['test_images_path']
        self.__sample_submission = params['sample_submission']

        self.__annotation_list_train: List[List]
        self.__aspect_ratio_pic_all: List[float]
        self.__dataset: Tuple[tf.data.Dataset, int]

        self.__input_height = params['input_height']
        self.__input_width = params['input_width']
        self.__training_ratio = params['training_ratio']
        self.batch_size = params['batch_size']

    def get_dataset_labels(self) -> List[float]:
        return [el[1] for el in self.__annotation_list_train_area]

    def generate_dataset(self):
        # Generate a list of list where each row represent an image and the list of the characters within it
        # (codified as integers) with relative coordinates of bbox
        self.__annotate()

        # Compute the average bbox ratio w.r.t. the image area considering all the images
        self.__annotate_char_area()

        # Generate the tf.data.Dataset containing all the objects
        self.__compose_dataset_object()

    def __annotate(self):
        df_train = pd.read_csv(self.__train_csv_path)

        # Remove any row with at least one nan value
        df_train = df_train.dropna(axis=0, how='any')
        df_train = df_train.reset_index(drop=True)

        annotation_list_train = []
        category_names = set()

        for i in range(len(df_train)):
            # Get one row for each label character for image i,
            # as: <category> <x> <y> <width> <height>
            ann = np.array(df_train.loc[i, "labels"].split(" ")).reshape(-1, 5)

            # Get set of unique categories
            category_names = category_names.union({i for i in ann[:, 0]})

        category_names = sorted(category_names)

        # Make a dict assigning an integer to each category
        dict_cat = {list(category_names)[j]: str(j) for j in range(len(category_names))}
        # inv_dict_cat = {str(j): list(category_names)[j] for j in range(len(category_names))}

        for i in range(len(df_train)):
            # Get one row for each label character for image i, as category,x,y,width,height
            ann = np.array(df_train.loc[i, "labels"].split(" ")).reshape(-1, 5)

            # Iterate over categories in first column of ann (characters)
            for j, category_name in enumerate(ann[:, 0]):
                # Change categories in integer values
                ann[j, 0] = int(dict_cat[category_name])

            # Calculate the center of each bbox
            # ann[:, 0] = class
            # ann[:, 1] = xmin
            # ann[:, 2] = ymin
            # ann[:, 3] = x width
            # ann[:, 4] = y height
            ann = ann.astype('int32')
            ann[:, 1] += ann[:, 3] // 2  # center_x
            ann[:, 2] += ann[:, 4] // 2  # center_y
            # ann[:, 0] = class
            # ann[:, 1] = x_center
            # ann[:, 2] = y_center
            # ann[:, 3] = x width
            # ann[:, 4] = y height

            annotation_list_train.append(
                ["{}/{}.jpg".format(self.__train_images_path, df_train.loc[i, "image_id"]),
                 ann])

            # print("Sample image show")
            # img = np.asarray(
            #    Image.open(annotation_list_train[0][0]).resize((self.__input_width,
            #       self.__input_height)).convert('RGB'))
            # plt.imshow(img)
            # plt.show()

        # This is a list of list where each row represent an image and the list of the characters within it
        # (codified as integers) with relative coordinates of bbox
        self.__annotation_list_train = annotation_list_train

    def __annotate_char_area(self) -> List[List]:
        """
        Computes the average bbox ratio w.r.t. the image area considering all the images,
        and plots a graph with ratio distribution.

        :return: a list where each row represent the average bbox ratio for character in an image.
        """
        self.__aspect_ratio_pic_all = []
        aspect_ratio_pic_all_test = []
        average_letter_size_all = []
        annotation_list_train_area = []
        # resize_dir = "resized/"

        # if os.path.exists(resize_dir) == False: os.mkdir(resize_dir)
        for i in range(len(self.__annotation_list_train)):
            with Image.open(self.__annotation_list_train[i][0]) as f:
                # Image dimensions
                width, height = f.size

                # Image area
                area = width * height
                aspect_ratio_pic = height / width
                self.__aspect_ratio_pic_all.append(aspect_ratio_pic)

                # Bbox area for each character in image (width * height)
                letter_size = self.__annotation_list_train[i][1][:, 3] * \
                              self.__annotation_list_train[i][1][:, 4]

                # List of ratios for each character
                letter_size_ratio = letter_size / area

                # Take the mean and append to list
                average_letter_size = np.mean(letter_size_ratio)
                average_letter_size_all.append(average_letter_size)

                # Add example for training with image path and log average bbox size for objects in it
                annotation_list_train_area.append([self.__annotation_list_train[i][0],
                                                   np.log(average_letter_size)])

        # Create test set
        # test_df = pd.read_csv(self.__sample_submission)
        # test_images = []
        # for image_id in test_df['image_id']:
        #     test_images.append(os.path.join(self.__test_images_path, image_id + '.jpg'))
        #
        # for i in range(len(test_images)):
        #     with Image.open(test_images[i]) as f:
        #         width, height = f.size
        #         aspect_ratio_pic = height / width
        #         aspect_ratio_pic_all_test.append(aspect_ratio_pic)

        plt.hist(np.log(average_letter_size_all), bins=100)
        plt.title('log(ratio of letter_size / picture_size)', loc='center', fontsize=12)
        # plt.show()

        self.__annotation_list_train_area = annotation_list_train_area

        return annotation_list_train_area  # List[str, float]

    def annotate_split_recommend(self, annotations_w_area: List[float]) -> List[List]:
        """
        Given a list of bbox sizes for each train image, computes the best size according to the image must be split

        :param annotations_w_area: list of predicted bbox areas for all characters
        :return: extended annotation list with recommended splits in format:
            image path: str, annotations: np.array, height split: float, width split: float
        """
        base_detect_num_h, base_detect_num_w = 25, 25

        annotation_list_train_w_split = []

        # For each predicted bbox size calculate recommended height and width
        for i, predicted_size in enumerate(annotations_w_area):
            # __aspect_ratio_pic_all = height / width
            detect_num_h = self.__aspect_ratio_pic_all[i] * exp(-predicted_size / 2)
            detect_num_w = exp(-predicted_size / 2)

            h_split_recommend = max([1, detect_num_h / base_detect_num_h])
            w_split_recommend = max([1, detect_num_w / base_detect_num_w])

            annotation_list_train_w_split.append(
                [self.__annotation_list_train[i][0], self.__annotation_list_train[i][1],
                 h_split_recommend,
                 w_split_recommend])
            # Format: image path, annotations, height split, width split

        # Just for test
        # for i in np.arange(0, 1):
        #     print("recommended height split:{}, recommended width_split:{}".format(
        #         annotation_list_train_w_split[i][2], annotation_list_train_w_split[i][3]))
        #     img = np.asarray(Image.open(annotation_list_train_w_split[i][0]).convert('RGB'))
        #     plt.imshow(img)
        #     plt.show()

        return annotation_list_train_w_split

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

        image_paths = [sample[0] for sample in self.__annotation_list_train_area]  # sample path
        image_labels = [sample[1] for sample in self.__annotation_list_train_area]  # avg bbox ratios

        image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        label_dataset = tf.data.Dataset.from_tensor_slices(image_labels)

        self.__dataset = (
            tf.data.Dataset.zip((image_dataset, label_dataset)).shuffle(buffer_size=500),
            len(image_paths))

    def get_training_set(self) -> Tuple[tf.data.Dataset, int]:
        TRAIN_SIZE = int(self.__training_ratio * self.__dataset[1])

        return (self.__dataset[0]
                .take(TRAIN_SIZE)
                .map(lambda path, label: tf.py_function(self.__preprocess_image,
                                                        [path, label, True, True],
                                                        (tf.float32, tf.float64)),
                     num_parallel_calls=AUTOTUNE)
                .batch(self.batch_size)
                .repeat()
                .prefetch(AUTOTUNE),
                TRAIN_SIZE)

    def get_validation_set(self) -> Tuple[tf.data.Dataset, int]:
        TRAIN_SIZE = int(self.__training_ratio * self.__dataset[1])

        return (self.__dataset[0]
                .skip(TRAIN_SIZE)
                .map(lambda path, label: tf.py_function(self.__preprocess_image,
                                                        [path, label, False, False],
                                                        (tf.float32, tf.float64)),
                     num_parallel_calls=AUTOTUNE)
                .batch(self.batch_size)
                .repeat()
                .prefetch(AUTOTUNE),
                self.__dataset[1] - TRAIN_SIZE)

    def get_test_set(self) -> Tuple[tf.data.Dataset, int]:
        # TODO
        return (None, 0)
