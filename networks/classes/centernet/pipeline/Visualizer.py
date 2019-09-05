import os

import pandas as pd
import regex as re

from networks.classes.centernet.utils.BBoxesVisualizer import BBoxesVisualizer


class Visualizer:

    def __init__(self, log):
        self.__log = log

    def visualize(self, max_visualizations=5):

        self.__log.info('Visualizing final results...')

        # Read the submission data from csv file
        path_to_submission = os.path.join('datasets', 'submission.csv')
        try:
            submission = pd.read_csv(path_to_submission, usecols=['image_id', 'labels'])
        except FileNotFoundError:
            raise Exception(
                'Cannot fetch data for visualization because no submission was written at {}\n'
                'Probably predict_on_test param was set to False, thus no submission has been written'
                    .format(path_to_submission))

        # Read the test data from csv file
        path_to_test_list = os.path.join('datasets', 'test_list.csv')
        try:
            test_list = pd.read_csv(path_to_test_list, usecols=['original_image', 'cropped_images', 'bboxes'])
        except FileNotFoundError:
            raise Exception(
                'Cannot fetch data for visualization because no submission was written at {}\n'
                'Probably predict_on_test param was set to False, thus no prediction has been made on test'
                    .format(path_to_test_list))

        # Initialize a bboxes visualizer object to print bboxes on images
        bbox_visualizer = BBoxesVisualizer(path_to_images=os.path.join('datasets', 'kaggle', 'testing', 'images'))

        # i counts the number of images that can be visualized
        i = 0

        submission_rows = [r for _, r in submission.iterrows()]
        test_data_rows = [r for _, r in test_list.iterrows()]

        # Iterate over the images
        for sub_data, test_data in zip(submission_rows, test_data_rows):

            if i == max_visualizations:
                break

            classes = [label.strip().split(' ')[0] for label in
                       re.findall(r"(?:\S*\s){3}", sub_data['labels'])]
            bboxes = test_data['bboxes'].split(' ')

            # Iterate over the predicted classes and corresponding bboxes
            labels = []
            for char_class, bbox in zip(classes, bboxes):
                ymin, xmin, ymax, xmax = bbox.split(':')

                xmin = round(float(xmin))
                ymin = round(float(ymin))
                xmax = round(float(xmax))
                ymax = round(float(ymax))

                labels.append([char_class,
                               xmin,
                               ymin,
                               xmax - xmin,
                               ymax - ymin])

            img_id = sub_data['image_id']
            self.__log.info('Visualizing image {}'.format(img_id))
            bbox_visualizer.visualize_bboxes(image_id=img_id, labels=labels)

            i += 1
