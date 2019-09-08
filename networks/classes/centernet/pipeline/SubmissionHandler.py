import os
from itertools import islice
from typing import Generator, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import regex as re

from networks.classes.centernet.utils.BBoxesVisualizer import BBoxesVisualizer


class SubmissionHandler:

    def __init__(self, dict_cat, log):
        self.__log = log
        self.__dict_cat = dict_cat

    def test(self, max_visualizations=5):

        self.__log.info('Testing the submission...')

        # Read the submission data from csv file
        path_to_submission = os.path.join('datasets', 'submission.csv')
        try:
            submission = pd.read_csv(path_to_submission, usecols=['image_id', 'labels'])
        except FileNotFoundError:
            raise Exception(
                'Cannot fetch data for visualization because no submission was written at {}\n'
                'Probably predict_on_test param was set to False, thus no submission has been written'
                    .format(path_to_submission))

        # Initialize a bboxes visualizer object to print bboxes on images
        bbox_visualizer = BBoxesVisualizer(path_to_images=os.path.join('datasets', 'kaggle', 'testing', 'images'))

        # i counts the number of images that can be visualized
        i = 0

        # Iterate over the images
        for _, sub_data in submission.iterrows():

            if i == max_visualizations:
                break

            labels = [label.strip().split(' ') for label in re.findall(r"(?:\s?\S*\s){2}\S*", sub_data['labels'])]
            labels = [[label[0], int(label[1]), int(label[2]), 5, 5] for label in labels]

            img_id = sub_data['image_id']
            self.__log.info('Visualizing image {}'.format(img_id))
            bbox_visualizer.visualize_bboxes(image_id=img_id, labels=labels)

            i += 1

    def __get_class(self, prediction: List):
        """
        Gets the unicode class from the predictions
        :param prediction: a list of class predictions
        :return:
        """

        class_index = np.argmax(prediction)
        return list(self.__dict_cat.keys())[list(self.__dict_cat.values()).index(class_index)]

    @staticmethod
    def __get_center_coords(bbox: str):
        """
        Gets the coordinates of the center of the bbox
        :param bbox: a string representing the coordinates of the bbox
        :return:
        """

        # Get the coordinates of the bbox
        ymin, xmin, ymax, xmax = bbox.split(':')

        ymin = round(float(ymin))
        xmin = round(float(xmin))
        ymax = round(float(ymax))
        xmax = round(float(xmax))

        return str(xmin + ((xmax - xmin) // 2)), str(ymin + ((ymax - ymin) // 2))

    def __write_img_with_chars(self, images_data, predictions_gen, num_iterations, path_to_submission):

        # Iterate over all the predicted original images
        for img_data in tqdm(images_data, total=num_iterations):

            cropped_images = list(img_data['cropped_images'].split(' '))
            bboxes = list(img_data['bboxes'].split(' '))
            labels = []

            for cropped_image, bbox in zip(cropped_images, bboxes):

                # Get prediction from generator
                try:
                    prediction = next(predictions_gen)
                except StopIteration:
                    break

                # Get the unicode class from the predictions
                unicode = self.__get_class(prediction)

                # Get the coordinates of the center of the bbox
                x, y = self.__get_center_coords(bbox)

                # Append the current label to the list of the labels of the current image
                labels.append(' '.join([unicode, x, y]))

            # Gather the data for the submission of the current image
            img_submission = pd.DataFrame(data={'image_id': img_data['original_image'],
                                                'labels': ' '.join(labels)},
                                          columns=['image_id', 'labels'],
                                          index=[0])

            # Write the submission to csv
            img_submission.to_csv(path_to_submission, mode='a', header=False)

            del img_submission
            del labels

    @staticmethod
    def __write_img_with_no_chars(path_to_submission):

        submission = pd.read_csv(path_to_submission)
        submitted_images = submission['image_id'].tolist()

        for img_path in tqdm(os.listdir(os.path.join('datasets', 'kaggle', 'testing', 'images'))):

            img_id = img_path.split(os.sep)[-1].split('.')[0]

            if img_id not in submitted_images:
                # Gather the data for the submission of the empty image
                img_submission = pd.DataFrame(data={'image_id': img_id,
                                                    'labels': ''},
                                              columns=['image_id', 'labels'],
                                              index=[0])

                # Write the submission to csv
                img_submission.to_csv(path_to_submission, mode='a', header=False)

    @staticmethod
    def __fetch_images_data(path_to_submission, test_list):

        # Delete the previous submission
        if os.path.isfile(path_to_submission):
            partial_sub = pd.read_csv(path_to_submission, usecols=['image_id', 'labels'])

            # Start iterating through the images from the last image inserted in the partial submission
            images_data = [img_data for _, img_data in islice(test_list.iterrows(), len(partial_sub.index) - 1, None)]

            # Remove last row (since it may be partial)
            partial_sub.drop(partial_sub.tail(1).index, inplace=True)
            partial_sub.to_csv(path_to_submission)

        else:
            # Start iterating through the images from the beginning
            images_data = [img_data for _, img_data in test_list.iterrows()]

            # Write the header
            pd.DataFrame(columns=['image_id', 'labels']).to_csv(path_to_submission)

        return images_data

    def write(self, predictions_gen: Generator):
        """
        Writes a submission csv file in the format:
        - names of columns : image_id, labels
        - example of row   : image_id, {label X Y} {...}
        :param predictions_gen: a list of class predictions for the cropped characters
        """

        self.__log.info('Writing submission data...')

        # Read the test data from csv file
        path_to_test_list = os.path.join('datasets', 'test_list.csv')
        try:
            test_list = pd.read_csv(path_to_test_list, usecols=['original_image', 'cropped_images', 'bboxes'])
        except FileNotFoundError:
            raise Exception('Cannot write submission because non test list was written at {}\n'
                            'Probably predict_on_test param was set to False, thus no prediction has been made on test'
                            .format(path_to_test_list))

        # Set the path to the submission
        path_to_submission = os.path.join('datasets', 'submission.csv')

        # Fetch the data of the images
        images_data = self.__fetch_images_data(path_to_submission, test_list)

        self.__log.info('Writing images with characters...')
        self.__write_img_with_chars(images_data=images_data,
                                    predictions_gen=predictions_gen,
                                    num_iterations=len(test_list.index),
                                    path_to_submission=path_to_submission)

        self.__log.info('Writing images with no characters...')
        self.__write_img_with_no_chars(path_to_submission)

        self.__log.info('Written submission data at {}'.format(path_to_submission))
