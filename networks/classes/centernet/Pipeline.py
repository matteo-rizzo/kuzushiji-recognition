import os
import shutil
import sys
from typing import List, Dict, Union, Tuple
import regex as re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from networks.classes.centernet.datasets.ClassificationDataset import ClassificationDataset
from networks.classes.centernet.datasets.DetectionDataset import DetectionDataset
from networks.classes.centernet.datasets.PreprocessingDataset import PreprocessingDataset
from networks.classes.centernet.models.HourglassNetwork import HourglassNetwork
from networks.classes.centernet.models.ModelCenterNet import ModelCenterNet
from networks.classes.centernet.utils.BBoxesHandler import BBoxesHandler
from networks.classes.centernet.utils.BBoxesVisualizer import BBoxesVisualizer
from networks.classes.centernet.utils.ImageCropper import ImageCropper
from networks.classes.centernet.utils.LossFunctionsGenerator import LossFunctionsGenerator


class CenterNetPipeline:

    def __init__(self, dataset_params: Dict, logs):
        self.__logs = logs
        self.__model_utils = ModelCenterNet(logs=self.__logs)
        self.__img_cropper = ImageCropper(log=self.__logs['execution'])
        self.__bb_handler = BBoxesHandler()
        self.__loss = LossFunctionsGenerator()

        self.__dataset_params = dataset_params
        self.__input_shape = (None, None, None)
        self.__dict_cat: Dict[str, int] = {}

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

                user_ok = True if user_input in ['y', 'Y', 'yes', 'ok'] else False

                if user_ok:
                    shutil.rmtree(folder)
                    self.__logs['execution'].info('Deleted weights checkpoint folder!')
                else:
                    self.__logs['execution'].info('Aborting after user command!')
                    sys.exit(0)

    def __set_input_shape(self, model_params):

        self.__input_shape = (model_params['input_width'],
                              model_params['input_height'],
                              model_params['input_channels'])

    def __resize_fn(self, path: str):
        """
        Utility function for image resizing

        :param path: the path to the image to be resized
        :return: a resized image
        """

        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (self.__input_shape[1], self.__input_shape[0]))

        return image_resized / 255

    def run_preprocessing(self, model_params: Dict, weights_path: str) -> PreprocessingDataset:
        """
        Creates and runs a CNN which takes an image/page of manuscript as input and predicts the
        average dimensional ratio between the characters and the image itself

        :param model_params: the parameters related to the network
        :param weights_path: the path to the saved weights (if present)
        :return: a ratio predictor
        """

        # Add dataset params to model params for simplicity
        model_params.update(self.__dataset_params)

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        self.__logs['execution'].info('Preprocessing the data...')

        # Build dataset for the preprocessing model
        dataset_avg_size = PreprocessingDataset(model_params)
        dataset_avg_size.generate_dataset()

        # Dictionary that map each char category into an integer value
        self.__dict_cat = dataset_avg_size.get_categories_dict()

        return dataset_avg_size

    def run_hourglass_detection(self, model_params, dataset_avg_size, weights_path, run_id):

        self.__logs['execution'].info('Initializing Hourglass model...')

        # Add dataset params to model params for simplicity
        model_params.update(self.__dataset_params)

        avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
        train_list: List[List] = dataset_avg_size.get_recommended_splits(avg_sizes)

        model = HourglassNetwork(run_id=run_id,
                                 log=self.__logs['training'],
                                 model_params=model_params,
                                 num_classes=6,
                                 num_stacks=1,
                                 num_channels=256,
                                 in_res=(int(model_params['input_width']),
                                         int(model_params['input_height'])),
                                 out_res=(int(model_params['output_width']),
                                          int(model_params['output_height'])))

        self.__logs['execution'].info('Hourglass model successfully initialized!')

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
        :return: a couple of lists with train and bbox data. Bbox data are available only if
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

        model_params.update(self.__dataset_params)
        self.__set_input_shape(model_params)

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        # Generate the CenterNet model
        model = self.__model_utils.build_model(input_shape=self.__input_shape,
                                               mode='detection',
                                               n_category=1)

        # Initialize the decay, if defined
        try:
            decay = float(model_params['decay'])
        except ValueError:
            decay = None

        # ---- MODEL COMPILING and WEIGHTS RESTORATION ----

        model.compile(optimizer=Adam(lr=model_params['learning_rate'],
                                     decay=decay if decay else 0.0),
                      loss=self.__loss.all_loss,
                      metrics=[self.__loss.size_loss,
                               self.__loss.heatmap_loss,
                               self.__loss.offset_loss])

        # Restore the saved weights, if required
        if model_params['restore_weights']:
            self.__model_utils.restore_weights(model=model,
                                               init_epoch=model_params['initial_epoch'],
                                               weights_folder_path=weights_path)

        # ---- DATASET CREATION ----

        # Get labels from dataset and compute the recommended split,
        # format is: [ [image path, annotations, height split, width split] ]
        avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
        train_list: List[List] = dataset_avg_size.get_recommended_splits(avg_sizes)

        # Generate the dataset for detection
        dataset_detection = DetectionDataset(model_params)

        # Pass the list of test images if we are in test mode,
        # otherwise pass None, so that the test set will not be generated
        test_list = self.__test_list if model_params['predict_on_test'] else None
        xy_train, xy_val, xy_eval = dataset_detection.generate_dataset(train_list, test_list)
        detection_ts, detection_ts_size = dataset_detection.get_training_set()
        detection_vs, detection_vs_size = dataset_detection.get_validation_set()
        detection_es, detection_es_size = dataset_detection.get_evaluation_set()
        detection_ps, detection_ps_size = dataset_detection.get_test_set()

        # ---- TRAINING ----

        # Train the model
        if model_params['train']:
            self.__logs['execution'].info(
                'Starting the training procedure for the object detection model...')

            # Set up the callbacks
            callbacks = self.__model_utils.setup_callbacks(weights_log_path=weights_path,
                                                           batch_size=model_params['batch_size'])

            # Start the training procedure
            self.__model_utils.train(model=model,
                                     init_epoch=model_params['initial_epoch'],
                                     epochs=model_params['epochs'],
                                     training_set=detection_ts,
                                     validation_set=detection_vs,
                                     training_steps=int(
                                         detection_ts_size // model_params['batch_size']) + 1,
                                     validation_steps=int(
                                         detection_vs_size // model_params['batch_size']) + 1,
                                     callbacks=callbacks)

        # ---- EVALUATION ----

        if model_params['evaluate']:

            # Evaluate the model
            self.__logs['test'].info('Evaluating the model...')
            metrics = self.__model_utils.evaluate(model=model,
                                                  evaluation_set=detection_es,
                                                  evaluation_steps=int(
                                                      detection_es_size // model_params[
                                                          'batch_size']) + 1)

            self.__logs['test'].info('Evaluation metrics:\n'
                                     'all_loss     : {}\n'
                                     'size_loss    : {}\n'
                                     'heatmap_loss : {}\n'
                                     'offset_loss  : {}'
                                     .format(metrics[0],
                                             metrics[1],
                                             metrics[2],
                                             metrics[3]))

            if model_params['show_prediction_examples']:
                self.__logs['test'].info('Showing prediction examples...')
                # Prepare a test dataset from the evaluation set taking its first 10 values
                test_path_list = [ann[0] for ann in xy_eval[:10]]
                mini_test = tf.data.Dataset.from_tensor_slices(test_path_list) \
                    .map(self.__resize_fn,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                    .batch(1) \
                    .prefetch(tf.data.experimental.AUTOTUNE)

                # Perform the prediction on the newly created dataset and show images
                detected_predictions = self.__model_utils.predict(model, mini_test)
                self.__bb_handler.get_bboxes(detected_predictions,
                                             mode='train',
                                             annotation_list=xy_eval[:10],
                                             show=True)

        # ---- GENERATION OF TEST PREDICTIONS ----

        predicted_test_bboxes: Union[Dict[str, np.ndarray], None] = None

        if model_params['predict_on_test']:
            self.__logs['execution'].info('Predicting test bounding boxes (takes time)...')
            test_predictions = self.__model_utils.predict(model=model,
                                                          dataset=detection_ps)
            self.__logs['execution'].info('Completed.')

            self.__logs['execution'].info('Converting test predictions into bounding boxes...')
            predicted_test_bboxes = self.__bb_handler.get_bboxes(test_predictions,
                                                                 mode='test',
                                                                 test_images_path=self.__test_list,
                                                                 show=False)
            self.__logs['execution'].info('Completed.')

        return train_list, predicted_test_bboxes

    def run_classification(self,
                           model_params: Dict,
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

        model_params.update(self.__dataset_params)
        self.__set_input_shape(model_params)

        # Check weights folder is not full of previous stuff
        if model_params['train'] and not model_params['restore_weights']:
            self.__check_no_weights_in_run_folder(weights_path)

        # Generate a model
        self.__model_utils = ModelCenterNet(logs=self.__logs)
        model = self.__model_utils.build_model(input_shape=self.__input_shape,
                                               mode='classification',
                                               n_category=len(self.__dict_cat))

        # Restore the weights, if required
        if model_params['restore_weights']:
            self.__model_utils.restore_weights(model=model,
                                               init_epoch=model_params['initial_epoch'],
                                               weights_folder_path=weights_path)

        # Initialize the decay, if present
        try:
            decay = float(model_params['decay'])
        except ValueError:
            decay = None

        # Compile the model
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=Adam(lr=model_params['learning_rate'],
                                     decay=decay if decay else 0.0),
                      metrics=["sparse_categorical_accuracy"])

        # Generate dataset object for the classification model
        crop_char_path_train = os.path.join('datasets', 'char_cropped_train')
        crop_char_path_test = os.path.join('datasets', 'char_cropped_test')

        # Train mode cropping
        if model_params['regenerate_crops_train']:
            train_list = self.__img_cropper.regenerate_crops_train(train_list, crop_char_path_train)
        else:
            train_list: List[Tuple[str, int]] = self.__img_cropper.load_crop_characters(crop_char_path_train,
                                                                                        mode='train')

        # Now 'train_list' is a list[(image_path, char_class)]

        # Test mode cropping
        test_list: Union[List[str], None] = None
        if model_params['predict_on_test']:
            if model_params['regenerate_crops_test']:
                test_list = self.__img_cropper.regenerate_crops_test(bbox_predictions,
                                                                     crop_char_path_test)
            else:
                test_list = self.__img_cropper.load_crop_characters(crop_char_path_test, mode='test')

            self.write_test_list_to_csv(test_list, bbox_predictions)

        # Now 'test_list' is a list[image_path] to cropped test images, or None if we are not in predict mode

        batch_size = int(model_params['batch_size'])
        dataset_classification = ClassificationDataset(model_params)

        # We need to pass it the training list, and the list of cropped images from test set
        # if we are in predict mode (otw pass we pass test_list=None).
        _, _, xy_eval = dataset_classification.generate_dataset(train_list, test_list)
        classification_ts, classification_ts_size = dataset_classification.get_training_set()
        classification_vs, classification_vs_size = dataset_classification.get_validation_set()
        classification_es, classification_es_size = dataset_classification.get_evaluation_set()
        classification_ps, classification_ps_size = dataset_classification.get_test_set()

        if model_params['train']:
            self.__logs['execution'].info(
                'Starting the training procedure for the classification model...')

            callbacks = self.__model_utils.setup_callbacks(weights_log_path=weights_path,
                                                           batch_size=batch_size)

            self.__model_utils.train(model=model,
                                     init_epoch=model_params['initial_epoch'],
                                     epochs=model_params['epochs'],
                                     training_set=classification_ts,
                                     validation_set=classification_vs,
                                     training_steps=int(classification_ts_size // batch_size) + 1,
                                     validation_steps=int(classification_vs_size // batch_size) + 1,
                                     callbacks=callbacks)

        if model_params['evaluate']:
            self.__logs['execution'].info('Evaluating the classification model...')

            metrics = self.__model_utils.evaluate(model=model,
                                                  evaluation_set=classification_es,
                                                  evaluation_steps=int(
                                                      classification_es_size // model_params[
                                                          'batch_size']) + 1)

            self.__logs['test'].info('Evaluation metrics:\n'
                                     'sparse_categorical_crossentropy : {}\n'
                                     'sparse_categorical_accuracy     : {}'
                                     .format(metrics[0], metrics[1]))

        if model_params['predict_on_test']:
            self.__logs['execution'].info('Starting the predict procedure of char class (takes much time)...')

            # Note that predictions.shape = (n_sample, n_category)
            predictions = self.__model_utils.predict(model=model, dataset=classification_ps)
            self.__logs['execution'].info('Prediction completed.')

            return predictions

        return None

    @staticmethod
    def write_test_list_to_csv(test_list: List, bbox_predictions: Dict):

        # Get all the names of the cropped images
        cropped_img_names = [cropped_img_path.split(os.sep)[-1] for cropped_img_path in test_list]

        # Get all the original names of the cropped images
        original_img_names = []
        for cropped_img_name, cropped_img_path in zip(cropped_img_names, test_list):
            original_img_name = '_'.join(cropped_img_name.split('_')[:-1])
            original_img_names.append(original_img_name)

        # Initialize a mapping <original image name> -> <list of cropped images names>
        original_img_to_cropped = {original_img_name: [] for original_img_name in original_img_names}

        # Initialize a mapping <cropped image name> -> <bbox coordinates>
        cropped_img_to_bbox = {cropped_img_name: None for cropped_img_name in cropped_img_names}

        # Initialize a mapping <original image name> -> <bbox coordinates>
        original_img_to_bbox = {original_img_name: [] for original_img_name in original_img_names}

        for cropped_img_name in cropped_img_names:
            # Map each original image to its cropped characters
            original_img_name = '_'.join(cropped_img_name.split('_')[:-1])
            original_img_to_cropped[original_img_name].append(cropped_img_name)

            # Map each cropped image to its bounding box
            cropped_img_id = int(cropped_img_name.split('_')[-1].split('.')[0])
            bbox_coords = [str(coord) for coord in bbox_predictions[original_img_name + '.jpg'][cropped_img_id][2:]]
            # Note that the coordinates are in format ymin:xmin:ymax:xmax
            cropped_img_to_bbox[cropped_img_name] = ':'.join(bbox_coords)

        for original_img_name, cropped_img_names in original_img_to_cropped.items():
            for cropped_img_name in cropped_img_names:
                original_img_to_bbox[original_img_name].append(cropped_img_to_bbox[cropped_img_name])

        original_img_to_bbox = {img_name: ' '.join(coords) for img_name, coords in original_img_to_bbox.items()}

        test_dict = {
            'original_image': [original_img_name for original_img_name in original_img_to_cropped.keys()],
            'cropped_images': [' '.join(cropped_img_names) for cropped_img_names in original_img_to_cropped.values()],
            'bboxes': [bbox for bbox in original_img_to_bbox.values()]
        }

        test_list_df = pd.DataFrame(test_dict)
        test_list_df.to_csv(os.path.join('datasets', 'test_list.csv'))

    def write_submission(self, predictions: List[List[float]]):
        """
        Writes a submission csv file in the format:
        - names of columns : image_id, labels
        - example of row   : image_id, {label X Y} {...}
        :param predictions: a list of class predictions for the cropped characters
        """

        # Initialize an empty dataset for submission
        submission = pd.DataFrame(columns=['image_id', 'labels'])

        # Initialize an empty dict with the data for the submission
        submission_dict = {}

        # Read the test data from csv file
        test_list = pd.read_csv(os.path.join('datasets', 'test_list.csv'),
                                usecols=['original_image', 'cropped_images', 'bboxes'])

        # Initialize an index to iterate over the predictions
        i = 0

        # Iterate over all the predicted original images
        for _, img_data in test_list.iterrows():

            cropped_images = list(img_data['cropped_images'].split(' '))
            bboxes = list(img_data['bboxes'].split(' '))

            for cropped_image, bbox in zip(cropped_images, bboxes):
                # Get the unicode class from the predictions
                prediction = predictions[i]
                class_index = np.where(prediction == max(prediction))[0][0]
                unicode = list(self.__dict_cat.keys())[list(self.__dict_cat.values()).index(class_index)]

                # Get the coordinates of the bbox
                ymin, xmin, ymax, xmax = bbox.split(':')

                ymin = round(float(ymin))
                xmin = round(float(xmin))
                ymax = round(float(ymax))
                xmax = round(float(xmax))

                x = str(xmin + ((xmax - xmin) // 2))
                y = str(ymin + ((ymax - ymin) // 2))

                # Append the current label to the list of the labels of the current images
                submission_dict.setdefault(img_data['original_image'], []).append(' '.join([unicode, x, y]))

                # Set the index for the next prediction
                i += 1

        # Convert the row in format: <image_id>, <label 1> <X_1> <Y_1> <label_2> <X_2> <Y_2> ...
        for original_image, labels in submission_dict.items():
            submission_dict[original_image] = ' '.join(labels)

        # Fill the dataframe with the data from the dict
        submission['image_id'] = submission_dict.keys()
        submission['labels'] = submission_dict.values()

        # Write the submission to csv
        submission.to_csv(os.path.join('datasets', 'submission.csv'))

    @staticmethod
    def visualize_final_results():

        # Read the submission data from csv file
        submission = pd.read_csv(os.path.join('datasets', 'submission.csv'), usecols=['image_id', 'labels'])

        # Read the test data from csv file
        test_list = pd.read_csv(os.path.join('datasets', 'test_list.csv'),
                                usecols=['original_image', 'cropped_images', 'bboxes'])

        # Initialize a bboxes visualizer object to print bboxes on images
        bbox_visualizer = BBoxesVisualizer(path_to_images=os.path.join('datasets', 'kaggle', 'testing', 'images'))

        submission_rows = [r for _, r in submission.iterrows()]
        test_data_rows = [r for _, r in test_list.iterrows()]

        # Iterate over the images
        for sub_data, test_data in zip(submission_rows, test_data_rows):
            classes = [label.strip().split(' ')[0] for label in re.findall(r"(?:\S*\s){3}", sub_data['labels'])]
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

            bbox_visualizer.visualize_bboxes(image_id=sub_data['image_id'], labels=labels)
