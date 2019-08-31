import os
import shutil
import sys
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from networks.classes.CenterNetClassificationDataset import CenterNetClassificationDataset
from networks.classes.CenterNetDetectionDataset import CenterNetDetectionDataset
from networks.classes.CenterNetPreprocessingDataset import CenterNetPreprocessingDataset
from networks.classes.HourglassNetwork import HourglassNetwork
from networks.classes.ModelCenterNet import ModelCenterNet
from networks.functions import losses
from networks.functions.bounding_boxes import get_bb_boxes
from networks.functions.cropping import load_crop_characters, annotations_to_bounding_boxes, \
    create_crop_characters_train, create_crop_characters_test, predictions_to_bounding_boxes


class CenterNetPipeline:
    __dict_cat: Dict[str, int]

    def __init__(self, dataset_params: Dict, logs):
        self.dataset_params = dataset_params
        self.logs = logs

        test_list = pd.read_csv(dataset_params['sample_submission'])['image_id'].to_list()
        base_path = os.path.join(os.getcwd(), 'datasets', 'kaggle', 'testing', 'images')
        self.__test_list = [str(os.path.join(base_path, img_id + '.jpg')) for img_id in test_list]

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
                confirmations = ['y', 'Y', 'yes', 'ok']

                user_ok = True if user_input in confirmations else False

                if user_ok:
                    shutil.rmtree(folder)
                    self.logs['execution'].info('Deleted weights checkpoint folder!')
                else:
                    self.logs['execution'].info('Aborting after user command!')
                    sys.exit(0)
        # If weights dir doesn't exists, no problem.

    def __resize_fn(self, path: str):
        """
        Utility function for image resizing

        :param path: the path to the image to be resized
        :return: a resized image
        """

        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (self.input_shape[1], self.input_shape[0]))

        return image_resized / 255

    def run_preprocessing(self, model_params: Dict, weights_path: str) -> CenterNetPreprocessingDataset:
        """
        Creates and runs a CNN which takes an image/page of manuscript as input and predicts the
        average dimensional ratio between the characters and the image itself

        :param model_params: the parameters related to the network
        :param weights_path: the path to the saved weights (if present)
        :return: a ratio predictor
        """

        # Add dataset params to model params for simplicity
        model_params.update(self.dataset_params)

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        self.logs['execution'].info('Preprocessing the data...')

        # Build dataset for model 1
        dataset_avg_size = CenterNetPreprocessingDataset(model_params)

        self.__dict_cat = dataset_avg_size.generate_dataset()
        # Dictionary that map each char category into an integer value

        size_check_ts, size_check_ts_size = dataset_avg_size.get_training_set()
        size_check_vs, size_check_vs_size = dataset_avg_size.get_validation_set()
        # size_check_ps, size_check_ps_size = dataset_avg_size.get_test_set()
        # input_shape = (model_params['input_width'], model_params['input_height'],
        #               model_params['input_channels])
        #
        # # Generate a model
        # model_utils = ModelUtilities()
        # model = model_utils.generate_model(input_shape=input_shape, mode=1)

        # try:
        #     decay = float(model_params['decay'])
        # except ValueError:
        #     decay = None
        #
        # model.compile(loss='mean_squared_error',
        #               optimizer=Adam(lr=model_params['learning_rate'],
        #                              decay=decay if decay else 0.0))
        #
        # # Restore the weights, if required
        # if model_params['restore_weights']:
        #     model_utils.restore_weights(model,
        #                                 self.logs['execution'],
        #                                 model_params['initial_epoch'],
        #                                 weights_path)
        #
        # # Train the model
        # if model_params['train']:
        #     self.logs['execution'].info('Starting the training procedure for model 1...')
        #
        #     # Set up the callbacks
        #     callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
        #                                             batch_size=model_params['batch_size'])
        #
        #     # Start the training procedure
        #     model_utils.train(model, self.logs['training'], model_params['initial_epoch'],
        #                       model_params['epochs'],
        #                       training_set=size_check_ts,
        #                       validation_set=size_check_vs,
        #                       training_steps=int(size_check_ts_size // model_params['batch_size'] + 1),
        #                       validation_steps=
        #                           int(size_check_vs_size // model_params['batch_size'] + 1),
        #                       callbacks=callbacks)
        #
        #     # Evaluate the model
        #     model_utils.evaluate(model, logger=self.logs['test'],
        #                          evaluation_set=size_check_vs,
        #                          evaluation_steps=
        #                               int(size_check_vs_size // model_params['batch_size'] + 1))

        return dataset_avg_size

    def run_hourglass_detection(self, model_params, dataset_avg_size, weights_path, run_id):

        self.logs['execution'].info('Initializing Hourglass model...')

        # Add dataset params to model params for simplicity
        model_params.update(self.dataset_params)

        avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
        train_list: List[List] = dataset_avg_size.annotate_split_recommend(avg_sizes)

        model = HourglassNetwork(run_id=run_id,
                                 log=self.logs['training'],
                                 model_params=model_params,
                                 num_classes=6,
                                 num_stacks=1,
                                 num_channels=256,
                                 in_res=(int(model_params['input_width']),
                                         int(model_params['input_height'])),
                                 out_res=(int(model_params['output_width']),
                                          int(model_params['output_height'])))

        self.logs['execution'].info('Hourglass model successfully initialized!')

        model.train(dataset_params=model_params,
                    train_list=train_list,
                    test_list=self.__test_list,
                    weights_path=weights_path)

    def run_detection(self,
                      model_params: Dict,
                      dataset_avg_size,
                      weights_path: str) -> (List[List], Union[Dict[str, np.ndarray], None]):
        """
        Creates and runs a CenterNet to perform the image detection

        :param model_params: the parameters related to the network
        :param dataset_avg_size: a ratio predictor
        :param weights_path: the path to the saved weights (if present)
        :return: a couple of list with train and bbox data. Bbox data are available only if
                 model_params['predict_on_test] is true. Otherwise return None

        Train list has the following structure:
            - train_list[0] = path to image
            - train_list[1] = annotations (ann)
            - train_list[2] = recommended height split
            - train_list[3] = recommended width split
            Where annotations is the bbox data:
            - ann[:, 1] = xmin
            - ann[:, 2] = ymin
            - ann[:, 3] = x width
            - ann[:, 4] = y height

        The bbox data consists of a list with the following structure (note that all are non numeric types):
         [<image_path>, <category>, <score>, <ymin>, <xmin>, <ymax>, <xmax>]

        The <category> value is always 0, because it is not the character category but the category of the center.
        """

        # Add dataset params to model params for simplicity
        model_params.update(self.dataset_params)

        input_shape = (
            model_params['input_width'], model_params['input_height'], model_params['input_channels']
        )

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        # Generate the CenterNet model
        model_utils = ModelCenterNet()
        model = model_utils.generate_model(input_shape=input_shape, n_category=1, mode=2)

        try:
            decay = float(model_params['decay'])
        except ValueError:
            decay = None

        model.compile(optimizer=Adam(lr=model_params['learning_rate'],
                                     decay=decay if decay else 0.0),
                      loss=losses.all_loss,
                      metrics=[losses.size_loss,
                               losses.heatmap_loss,
                               losses.offset_loss])

        # Restore the saved weights if required
        if model_params['restore_weights']:
            model_utils.restore_weights(model=model,
                                        logger=self.logs['execution'],
                                        init_epoch=model_params['initial_epoch'],
                                        weights_folder_path=weights_path)

        # plot_model(model, os.path.join(weights_path, os.pardir, 'plot.png'), show_shapes=True)

        # Get labels from dataset and compute the recommended split
        avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
        train_list: List[List] = dataset_avg_size.annotate_split_recommend(avg_sizes)
        # format: [ [image path, annotations, height split, width split] ]

        # Generate the dataset for detection
        self.dataset_params['batch_size'] = model_params['batch_size']
        self.dataset_params['batch_size_predict'] = model_params['batch_size_predict']
        dataset_detection = CenterNetDetectionDataset(model_params)

        # Pass the list of test images if we are in test mode, otw pass None, so that the test set
        # will not be generated.
        test_list = self.__test_list if model_params['predict_on_test'] else None
        xy_train, xy_val = dataset_detection.generate_dataset(train_list, test_list)
        detection_ts, detection_ts_size = dataset_detection.get_training_set()
        detection_vs, detection_vs_size = dataset_detection.get_validation_set()
        detection_ps, detection_ps_size = dataset_detection.get_test_set()

        # Train the model
        if model_params['train']:
            self.logs['execution'].info(
                'Starting the training procedure for the object detection model...')

            # Set up the callbacks
            callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
                                                    batch_size=model_params['batch_size'])

            # Start the training procedure
            model_utils.train(model=model,
                              logger=self.logs['training'],
                              init_epoch=model_params['initial_epoch'],
                              epochs=model_params['epochs'],
                              training_set=detection_ts,
                              validation_set=detection_vs,
                              training_steps=int(detection_ts_size // model_params['batch_size']) + 1,
                              validation_steps=int(detection_vs_size // model_params['batch_size']) + 1,
                              callbacks=callbacks)

            # Evaluate the model
            metrics = model_utils.evaluate(model=model,
                                           logger=self.logs['test'],
                                           evaluation_set=detection_vs,
                                           evaluation_steps=int(
                                               detection_vs_size // model_params['batch_size'] + 1))

            self.logs['test'].info('Evaluation metrics:\n'
                                   'all_loss     : {}\n'
                                   'size_loss    : {}\n'
                                   'heatmap_loss : {}\n'
                                   'offset_loss  : {}'
                                   .format(metrics[0],
                                           metrics[1],
                                           metrics[2],
                                           metrics[3]))

        # ---- MINI TEST ON VALIDATION SET ----

        if model_params['show_prediction_examples']:
            # Prepare a test dataset from the validation set taking its first 10 values
            test_path_list = [ann[0] for ann in xy_val[:10]]
            mini_test = tf.data.Dataset.from_tensor_slices(test_path_list) \
                .map(self.__resize_fn,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .batch(1) \
                .prefetch(tf.data.experimental.AUTOTUNE)

            # Perform the prediction on the newly created dataset and show images
            detected_predictions = model_utils.predict(model, self.logs['test'], mini_test)
            get_bb_boxes(detected_predictions,
                         mode='train',
                         annotation_list=xy_val[:10],
                         print=True)

        # ---- END MINI TEST ----

        predicted_test_bboxes: Union[Dict[str, np.ndarray], None] = None

        if model_params['predict_on_test']:
            self.logs['execution'].info('Predicting test bounding boxes (takes time)...')
            test_predictions = model_utils.predict(model=model,
                                                   logger=self.logs['test'],
                                                   dataset=detection_ps)

            self.logs['execution'].info('Completed.')

            self.logs['execution'].info('Converting test predictions into bounding boxes...')
            predicted_test_bboxes = get_bb_boxes(test_predictions,
                                                 mode='test',
                                                 test_images_path=self.__test_list,
                                                 print=False)
            self.logs['execution'].info('Completed.')

        return train_list, predicted_test_bboxes

    def run_classification(self, model_params: Dict,
                           train_list: List[List],
                           bbox_predictions: Union[Dict[str, np.ndarray], None],
                           weights_path: str):
        """
        Classifies each character according to the available classes via a CNN

        :param model_params: the parameters related to the network
        :param train_list: a train data list predicted at the object detection step
        :param bbox_predictions: the bbox data predicted at the object detection step or None if
                                predictions were not done.
        :param weights_path: the path to the saved weights (if present)
        :return: a couple of list with train and bbox data.
        """

        # Add dataset params to model params for simplicity
        model_params.update(self.dataset_params)

        input_shape = (model_params['input_width'],
                       model_params['input_height'],
                       model_params['input_channels'])

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        # Generate a model
        model_utils = ModelCenterNet()
        model = model_utils.generate_model(input_shape=input_shape,
                                           mode=3,
                                           n_category=len(self.__dict_cat))

        # Restore the weights, if required
        if model_params['restore_weights']:
            model_utils.restore_weights(model=model,
                                        logger=self.logs['execution'],
                                        init_epoch=model_params['initial_epoch'],
                                        weights_folder_path=weights_path)

        try:
            decay = float(model_params['decay'])
        except ValueError:
            decay = None

        # Compile the model
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(lr=model_params['learning_rate'],
                                     decay=decay if decay else 0.0),
                      metrics=["accuracy"])

        # Generate dataset object for model 3
        crop_char_path_train = os.path.join(os.getcwd(), 'datasets', 'char_cropped_train')
        crop_char_path_test = os.path.join(os.getcwd(), 'datasets', 'char_cropped_test')

        # Train mode
        if model_params['regenerate_crops_train']:
            # NOTE: the following 2 scripts are only run once to generate the images for training.
            self.logs['execution'].info(
                'Starting procedure to regenerate cropped train character images')

            self.logs['execution'].info('Getting bounding boxes from annotations...')
            crop_formatted_list = annotations_to_bounding_boxes(train_list)

            self.logs['execution'].info('Cropping images to characters...')
            train_list: List[Tuple[str, int]] \
                = create_crop_characters_train(crop_formatted_list, crop_char_path_train)
            self.logs['execution'].info('Cropping done successfully!')

        else:
            # Load from folder
            train_list: List[Tuple[str, int]] = load_crop_characters(crop_char_path_train, mode='train')

        # Test mode
        test_list: Union[List[str], None] = None
        if model_params['predict_on_test']:
            if model_params['regenerate_crops_test']:
                self.logs['execution'].info('Starting procedure to regenerate cropped test character images')

                # bbox_predictions is a dict: {image: np.arr[category, score, xmin, ymin, sxmax, ymax]}
                nice_formatted_dict: Dict[str, np.array] = predictions_to_bounding_boxes(bbox_predictions)

                self.logs['execution'].info('Cropping test images to characters...')
                test_list = create_crop_characters_test(nice_formatted_dict, crop_char_path_test)
                self.logs['execution'].info('Cropping done successfully!')

            else:
                # Load from folder
                test_list = load_crop_characters(crop_char_path_test, mode='test')

        # Now 'train_list' is a list[(image_path, char_class)]
        # Now 'test_list' is a list[image_path] to cropped test images, or
        # None if we are not in predict mode.
        # Now that we have the list in the correct format, let's generate together the tf.data.Dataset

        batch_size = int(model_params['batch_size'])
        dataset_classification = CenterNetClassificationDataset(model_params)

        # We need to pass it the training list, and the list of cropped images from test set if we are
        # in predict mode (otw pass we pass test_list=None).
        x_train, x_val = dataset_classification.generate_dataset(train_list, test_list)
        classification_ts, classification_ts_size = dataset_classification.get_training_set()
        classification_vs, classification_vs_size = dataset_classification.get_validation_set()
        classification_ps, classification_ps_size = dataset_classification.get_test_set()

        if model_params['train']:
            self.logs['execution'].info(
                'Starting the training procedure for the classification model...')

            callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
                                                    batch_size=batch_size)

            model_utils.train(model=model,
                              logger=self.logs['training'],
                              init_epoch=model_params['initial_epoch'],
                              epochs=model_params['epochs'],
                              training_set=classification_ts,
                              validation_set=classification_vs,
                              training_steps=int(classification_ts_size // batch_size) + 1,
                              validation_steps=int(classification_vs_size // batch_size) + 1,
                              callbacks=callbacks)

        if model_params['predict_on_test']:
            self.logs['execution'].info(
                'Starting the predict procedure of char class (takes much time)...')

            predictions = model_utils.predict(model=model,
                                              logger=self.logs['test'],
                                              dataset=classification_ps)

            self.logs['execution'].info('Prediction completed.')

            # predictions.shape = (batch, out_height, out_width, n_category)

            return predictions

        return None
