import csv
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
            test_list = pd.read_csv(path_to_test_list,
                                    usecols=['original_image', 'cropped_images', 'bboxes'])
        except FileNotFoundError:
            raise Exception('Cannot write submission because non test list was written at {}\n'
                            'Probably predict_on_test param was set to False, thus no prediction has been made on test'
                            .format(path_to_test_list))

        # Set the path to the submission
        path_to_submission = os.path.join('datasets', 'submission.csv')

        # Delete the previous submission
        if os.path.isfile(path_to_submission):
            # Remove last row (because may be partial)
            partial_submission = pd.read_csv(path_to_submission, usecols=['image_id', 'labels'])
            partial_submission.drop(partial_submission.tail(1).index, inplace=True)
            partial_submission.to_csv(path_to_submission)

            images_data = [img_data for _, img_data in islice(partial_submission.iterrows(), len(partial_submission)-1, None)]
        else:
            images_data = [img_data for _, img_data in test_list.iterrows()]

        # Write the header
        pd.DataFrame(columns=['image_id', 'labels']).to_csv(path_to_submission)

        self.__log.info('Writing images with characters...')

        # Iterate over all the predicted original images
        for img_data in tqdm(images_data, total=len(test_list.index)):

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

        self.__log.info('Writing images with no characters...')

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

        self.__log.info('Written submission data at {}'.format(path_to_submission))

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
