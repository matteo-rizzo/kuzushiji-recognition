from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# from tensorflow.python.keras.layers import


AUTOTUNE = tf.data.experimental.AUTOTUNE


class Preprocessor:
    def __init__(self, params: Dict):
        self.__train_csv_path = params.train_csv_path
        self.__train_images_path = params.train_images_path
        self.__test_images_path = params.test_images_path
        self.__sample_submission = params.sample_submission
        self.__annotation_list_train: List[List[str, pd.DataFrame]]
        self.__dataset: Tuple[tf.data.Dataset, int]
        self.__training_size = params.training_size

    def annotate(self):
        # train_csv_path = "datasets/kaggle/image_labels_map.csv"
        # train_images_path = "datasets/kaggle/training/images/"
        # path_3 = "../input/test_images/"
        # path_4 = "../input/sample_submission.csv"
        df_train = pd.read_csv(self.__train_csv_path)
        # print(df_train.head())
        # print(df_train.shape)

        # Remove any row with at least one nan value
        df_train = df_train.dropna(axis=0, how='any')
        df_train = df_train.reset_index(drop=True)
        # print(df_train.shape)

        annotation_list_train = []
        category_names = set()

        for i in range(len(df_train)):
            # Get one row for each label character for image i, as category,x,y,width,height
            ann = np.array(df_train.loc[i, "labels"].split(" ")).reshape(-1, 5)

            # Get set of unique categories
            category_names = category_names.union({i for i in ann[:, 0]})

        category_names = sorted(category_names)

        # Make a dict assigning an integer to each category
        dict_cat = {list(category_names)[j]: str(j) for j in range(len(category_names))}
        inv_dict_cat = {str(j): list(category_names)[j] for j in range(len(category_names))}
        # print(dict_cat)

        for i in range(len(df_train)):
            # Get one row for each label character for image i, as category,x,y,width,height
            ann = np.array(df_train.loc[i, "labels"].split(" ")).reshape(-1, 5)

            # Iterate over categories in first column of ann (characters)
            # Change categories in integer values
            for j, category_name in enumerate(ann[:, 0]):
                ann[j, 0] = int(dict_cat[category_name])

            # Calculate center for each bbox.
            # ann[:, 0] = class
            # ann[:, 1] = xmin
            # ann[:, 2] = ymin
            # ann[:, 3] = x width
            # ann[:, 4] = y height
            ann = ann.astype('int32')
            ann[:, 1] += ann[:, 3] // 2  # center_x
            ann[:, 2] += ann[:, 4] // 2  # center_y
            annotation_list_train.append(
                ["{}{}.jpg".format(self.__train_images_path, df_train.loc[i, "image_id"]), ann])

        print("Sample image show")
        input_width, input_height = 512, 512
        img = np.asarray(
            Image.open(annotation_list_train[0][0]).resize((input_width, input_height)).convert('RGB'))
        plt.imshow(img)
        plt.show()

    def __check_char_size(self) -> List[List[str, float]]:
        aspect_ratio_pic_all = []
        aspect_ratio_pic_all_test = []
        average_letter_size_all = []
        train_input_for_size_estimate = []
        # resize_dir = "resized/"

        # if os.path.exists(resize_dir) == False: os.mkdir(resize_dir)
        for i in range(len(self.__annotation_list_train)):
            with Image.open(self.__annotation_list_train[i][0]) as f:
                # Image dimensions
                width, height = f.size
                # Image area
                area = width * height
                aspect_ratio_pic = height / width
                aspect_ratio_pic_all.append(aspect_ratio_pic)

                # Bbox area for each character in image (width * height)
                letter_size = self.__annotation_list_train[i][1][:, 3] * \
                              self.__annotation_list_train[i][1][:, 4]
                # List of ratios for each character
                letter_size_ratio = letter_size / area

                # Take the mean and append to list
                average_letter_size = np.mean(letter_size_ratio)
                average_letter_size_all.append(average_letter_size)

                # Add example for training with image path and log average bbox size for objects in it
                train_input_for_size_estimate.append(
                    [self.__annotation_list_train[i][0], np.log(average_letter_size)])

        # Create test set
        test_df = pd.read_csv(self.__sample_submission)
        test_images = []
        for image_id in test_df['image_id']:
            test_images.append(os.path.join(self.__test_images_path, image_id + '.jpg'))

        for i in range(len(test_images)):
            with Image.open(test_images[i]) as f:
                width, height = f.size
                aspect_ratio_pic = height / width
                aspect_ratio_pic_all_test.append(aspect_ratio_pic)

        plt.hist(np.log(average_letter_size_all), bins=100)
        plt.title('log(ratio of letter_size to picture_size))', loc='center', fontsize=12)
        plt.show()

        return train_input_for_size_estimate

    def __preprocess_image(self, image, label, is_train=True, random_crop=True):
        input_width, input_height = 512, 512

        if random_crop:
            crop_ratio = np.random.uniform(0.7, 1)
        else:
            crop_ratio = 1

        # Load image
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_jpeg(image_string)

        # random crop
        if random_crop and is_train:

            # Get image size
            f = Image.open(image)
            pic_width, pic_height = f.size

            top_offset = np.random.randint(0, pic_height - int(crop_ratio * pic_height)) / (pic_height
                                                                                            - 1)
            left_offset = np.random.randint(0, pic_width - int(crop_ratio * pic_width)) / (pic_width - 1)
            bottom_offset = top_offset + int(crop_ratio * pic_height) / (pic_height - 1)
            right_offset = left_offset + int(crop_ratio * pic_width) / (pic_width - 1)

            image_resized = tf.image.crop_and_resize(image=[image_decoded],
                                                     box_ind=[0],
                                                     boxes=[[top_offset, left_offset, bottom_offset,
                                                             right_offset]],
                                                     crop_size=[input_width, input_height])

        else:
            image_resized = tf.image.resize_images(images=image_decoded,
                                                   size=[input_width, input_height])

        if random_crop and is_train:
            # Update average bbox size after cropping
            label -= np.log(crop_ratio)

        # Make sure values are in range [0, 255]
        image_resized /= 255

        return image_resized, label

    def dataset_gen(self):
        train_input = self.__check_char_size()

        image_paths = [sample[0] for sample in train_input]
        image_labels = [sample[1] for sample in train_input]

        image_dataset = tf.data.Dataset.from_sparse_tensor_slices(image_paths)

        label_dataset = tf.data.Dataset.from_sparse_tensor_slices(image_labels)

        self.__dataset = (
            tf.data.Dataset.zip((image_dataset, label_dataset)).shuffle(buffer_size=150),
            len(image_paths))

    def get_training_set(self) -> tf.data.Dataset:

        TRAIN_SIZE = self.__training_size * self.__dataset[1]

        return (self.__dataset[0]
                .take(TRAIN_SIZE)
                .map(lambda path, label: self.__preprocess_image(path, label,
                                                                 is_train=True,
                                                                 random_crop=True),
                     num_parallel_calls=AUTOTUNE)
                .batch(100)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_validation_set(self) -> tf.data.Dataset:

        TRAIN_SIZE = self.__training_size * self.__dataset[1]

        return (self.__dataset[0]
                .skip(TRAIN_SIZE)
                .map(lambda path, label: self.__preprocess_image(path, label,
                                                                 is_train=False,
                                                                 random_crop=True),
                     num_parallel_calls=AUTOTUNE)
                .batch(100)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_test_set(self) -> tf.data.Dataset:
        pass
