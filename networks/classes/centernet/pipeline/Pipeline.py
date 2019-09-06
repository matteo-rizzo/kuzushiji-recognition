import os
import shutil
import sys
from typing import List, Dict, Union, Generator
import numpy as np
from networks.classes.centernet.datasets.PreprocessingDataset import PreprocessingDataset
from networks.classes.centernet.pipeline.Preprocessor import Preprocessor
from networks.classes.centernet.pipeline.Detector import Detector
from networks.classes.centernet.pipeline.Classifier import Classifier
from networks.classes.centernet.pipeline.SubmissionHandler import SubmissionHandler
from networks.classes.centernet.pipeline.Visualizer import Visualizer
from networks.classes.general_utilities import Params


class CenterNetPipeline:

    def __init__(self, dataset_params: Dict, logs):
        self.__logs = logs
        self.__dataset_params = dataset_params
        self.__dict_cat: Dict[str, int] = {}

    def __check_no_weights_in_run_folder(self, folder: str):
        """
        Checks if the given folder contains weights files and  asks the user permission to delete them

        :param folder: the path to the folder to be checked
        """

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

    def __run_preprocessing(self, model_params: Dict, weights_path: str) -> PreprocessingDataset:
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

    def __run_detection(self,
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

    def __run_classification(self,
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
                                   bbox_predictions=bbox_predictions)

    def __write_submission(self, predictions_gen: Generator):
        """
        Writes a submission csv file in the format:
        - names of columns : image_id, labels
        - example of row   : image_id, {label X Y} {...}
        :param predictions_gen: a list of class predictions for the cropped characters
        """

        sub_writer = SubmissionHandler(dict_cat=self.__dict_cat, log=self.__logs['execution'])
        sub_writer.write(predictions_gen)
        sub_writer.test(max_visualizations=5)

    def __visualize_final_results(self, max_visualizations: int = 5):
        """
        Visualizes the predicted results

        :param max_visualizations: the maximum number of images that can be visualized
        """

        visualizer = Visualizer(log=self.__logs['execution'])
        visualizer.visualize(max_visualizations)

    def run_pipeline(self, operations: List, params: Params, experiment_path: str):
        """
        Runs the learning pipeline

        :param operations: a list containing one or more of the following values:
            - preprocessing
            - detection
            - classification
            - submission
            - visualization
        :param params: the parameters of the models
        :param experiment_path: the base path to the current experiment
        """

        self.__logs['execution'].info('Starting learning pipeline with operations: {}'.format(operations))

        # --- STEP 1: Pre-processing (Check Object Size) ---
        if 'preprocessing' in operations:
            dataset_avg_size = self.__run_preprocessing(model_params=params.model_1,
                                                        weights_path=os.path.join(experiment_path + '_1', 'weights'))

        # --- STEP 2: Detection by CenterNet ---
        if 'detection' in operations:

            if 'preprocessing' not in operations:
                raise Exception('ERROR: Cannot perform detection without preprocessing!'
                                'Please specify "preprocessing" in the list of operations')

            train_list, bbox_predictions = self.__run_detection(model_params=params.model_2,
                                                                dataset_avg_size=dataset_avg_size,
                                                                weights_path=os.path.join(experiment_path + '_2',
                                                                                          'weights'))

        # --- STEP 3: Classification ---
        if 'classification' in operations:

            if 'detection' not in operations:
                raise Exception('ERROR: Cannot perform classification without detection!'
                                'Please specify "detection" in the list of operations')

            predictions = self.__run_classification(model_params=params.model_3,
                                                    train_list=train_list,
                                                    bbox_predictions=bbox_predictions,
                                                    weights_path=os.path.join(experiment_path + '_3', 'weights'))

        # -- STEP 4:  Analysis and visualization of results ---
        if 'submission' in operations:

            if 'classification' not in operations:
                raise Exception('ERROR: Cannot write submission without classification!'
                                'Please specify "classification" in the list of operations')

            self.__write_submission(predictions)

        if 'visualization' in operations:
            self.__visualize_final_results()
