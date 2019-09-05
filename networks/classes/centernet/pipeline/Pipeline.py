import os
import shutil
import sys
from typing import List, Dict, Union, Generator
import numpy as np
from networks.classes.centernet.datasets.PreprocessingDataset import PreprocessingDataset
from networks.classes.centernet.pipeline.Preprocessor import Preprocessor
from networks.classes.centernet.pipeline.Detector import Detector
from networks.classes.centernet.pipeline.Classifier import Classifier
from networks.classes.centernet.pipeline.SubmissionWriter import SubmissionWriter
from networks.classes.centernet.pipeline.Visualizer import Visualizer


class CenterNetPipeline:

    def __init__(self, dataset_params: Dict, logs):
        self.__logs = logs
        self.__dataset_params = dataset_params
        self.__dict_cat: Dict[str, int] = {}

    def __check_no_weights_in_run_folder(self, folder: str):
        if os.path.isdir(folder):
            if len(os.listdir(folder)):
                user_input = input('WARNING! You want to train the network without restoring weights,\n'
                                   'but folder {} is not empty.\n'
                                   'It may contain tensorboard files and checkpoints from previous '
                                   'runs.\n'
                                   'Do you wish to delete all existing files in that folder and proceed '
                                   'with the training?\n'
                                   'Please refuse to abort the execution.\n'
                                   'Confirm? [Y/n]\n'.format(folder))

                user_ok = True if user_input in ['y', 'Y', 'yes', 'ok'] else False

                if user_ok:
                    shutil.rmtree(folder)
                    self.__logs['execution'].info('Deleted weights checkpoint folder!')
                else:
                    self.__logs['execution'].info('Aborting after user command!')
                    sys.exit(0)

    def run_preprocessing(self, model_params: Dict, weights_path: str) -> PreprocessingDataset:
        """
        Creates and runs a CNN which takes an image/page of manuscript as input and predicts the
        average dimensional ratio between the characters and the image itself

        :param model_params: the parameters related to the network
        :param weights_path: the path to the saved weights (if present)
        :return: a ratio predictor
        """

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        preprocessor = Preprocessor(dataset_params=self.__dataset_params, log=self.__logs['execution'])
        dataset_avg_size, self.__dict_cat = preprocessor.preprocess_data(model_params)

        return dataset_avg_size

    def run_detection(self,
                      model_params: Dict,
                      dataset_avg_size,
                      weights_path: str) -> (List[List], Union[Dict[str, np.ndarray], None]):
        """
        Creates and runs a CenterNet to perform the image detection

        :param model_params: the parameters related to the network
        :param dataset_avg_size: a ratio predictor
        :param weights_path: the path to the saved weights (if present)
        :return: a couple of lists with train and bbox data. Bbox data are available only if
                 model_params['predict_on_test] is true. Otherwise return None
        """

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        detector = Detector(model_params=model_params,
                            dataset_params=self.__dataset_params,
                            weights_path=weights_path,
                            logs=self.__logs)

        return detector.detect(dataset_avg_size)

    def run_classification(self,
                           model_params: Dict,
                           train_list: List[List],
                           bbox_predictions: Union[Dict[str, np.ndarray], None],
                           weights_path: str) -> Union[Generator, None]:
        """
        Classifies each character according to the available classes via a CNN

        :param model_params: the parameters related to the network
        :param train_list: a train data list predicted at the object detection step
        :param bbox_predictions: the bbox data predicted at the object detection step or
            None if predictions were not done.
        :param weights_path: the path to the saved weights (if present)
        :return: a couple of list with train and bbox data.
        """

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        classifier = Classifier(model_params=model_params,
                                dataset_params=self.__dataset_params,
                                weights_path=weights_path,
                                num_categories=len(self.__dict_cat),
                                logs=self.__logs)

        return classifier.classify(train_list=train_list,
                                   bbox_predictions=bbox_predictions, )

    def write_submission(self, predictions_gen: Generator):
        """
        Writes a submission csv file in the format:
        - names of columns : image_id, labels
        - example of row   : image_id, {label X Y} {...}
        :param predictions_gen: a list of class predictions for the cropped characters
        """

        sub_writer = SubmissionWriter(dict_cat=self.__dict_cat, log=self.__logs['execution'])
        sub_writer.write(predictions_gen)

    def visualize_final_results(self, max_visualizations: int = 5):
        """
        Visualizes the predicted results

        :param max_visualizations: the maximum number of images that can be visualized
        """

        visualizer = Visualizer(log=self.__logs['execution'])
        visualizer.visualize(max_visualizations)
