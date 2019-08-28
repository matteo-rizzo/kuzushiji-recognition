import os
from typing import List

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from networks.classes.CenterNetClassificationDataset import ClassifierDataset
from networks.classes.CenterNetDetectionDataset import CenterNetDataset
from networks.classes.ModelCenterNet import ModelUtilities
from networks.classes.SizePredictDataset import SizePredictDataset
from networks.functions import losses
from networks.functions.bounding_boxes import get_bb_boxes
from networks.functions.cropping import load_crop_characters, annotations_to_bounding_boxes, \
    create_crop_characters_train


class CenterNetPipeline:
    def __init__(self, dataset_params, input_shape, logs):
        self.dataset_params = dataset_params
        self.input_shape = input_shape
        self.logs = logs

    def __resize_fn(self, path):
        """
        Utility function for image resizing

        :param path: the path to the image to be resized
        :return: a resized image
        """

        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (self.input_shape[1], self.input_shape[0]))

        return image_resized / 255

    def run_preprocessing(self, model_params, weights_path) -> SizePredictDataset:
        """
        Creates and runs a CNN which takes an image/page of manuscript as input and predicts the
        average dimensional ratio between the characters and the image itself

        :param model_params: the parameters related to the network
        :param weights_path: the path to the saved weights (if present)
        :return: a ratio predictor
        """

        self.logs['execution'].info('Preprocessing the data...')

        # Build dataset for model 1
        self.dataset_params['batch_size'] = model_params['batch_size']
        dataset_avg_size = SizePredictDataset(self.dataset_params)

        dataset_avg_size.generate_dataset()

        size_check_ts, size_check_ts_size = dataset_avg_size.get_training_set()
        size_check_vs, size_check_vs_size = dataset_avg_size.get_validation_set()
        # size_check_ps, size_check_ps_size = dataset_avg_size.get_test_set()
        #
        # # Generate a model
        # model_utils = ModelUtilities()
        # model = model_utils.generate_model(input_shape=self.input_shape, mode=1)
        # model.compile(loss='mean_squared_error',
        #               optimizer=Adam(lr=model_params['learning_rate']))
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
        #     model_utils.train(model, self.logs['training'], model_params['initial_epoch'], model_params['epochs'],
        #                       training_set=size_check_ts,
        #                       validation_set=size_check_vs,
        #                       training_steps=int(size_check_ts_size // model_params['batch_size'] + 1),
        #                       validation_steps=int(size_check_vs_size // model_params['batch_size'] + 1),
        #                       callbacks=callbacks)
        #
        #     # Evaluate the model
        #     model_utils.evaluate(model, logger=self.logs['test'],
        #                          evaluation_set=size_check_vs,
        #                          evaluation_steps=int(size_check_vs_size // model_params['batch_size'] + 1))

        return dataset_avg_size

    def run_detection(self, model_params, dataset_avg_size, weights_path) -> (List, List):
        """
        Creates and runs a CenterNet to perform the image detection

        :param model_params: the parameters related to the network
        :param dataset_avg_size: a ratio predictor
        :param weights_path: the path to the saved weights (if present)
        :return: a couple of list with train and bbox data.

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

        # Generate the CenterNet model
        model_utils = ModelUtilities()
        model = model_utils.generate_model(input_shape=self.input_shape, mode=2)
        model.compile(optimizer=Adam(lr=model_params['learning_rate']),
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

        # Get labels from dataset and compute the recommended split
        avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
        train_list = dataset_avg_size.annotate_split_recommend(avg_sizes)

        # Generate the dataset for detection
        self.dataset_params['batch_size'] = model_params['batch_size']
        dataset_detection = CenterNetDataset(self.dataset_params)
        x_train, x_val = dataset_detection.generate_dataset(train_list)
        detection_ts, detection_ts_size = dataset_detection.get_training_set()
        detection_vs, detection_vs_size = dataset_detection.get_validation_set()

        # Train the model
        if model_params['train']:
            self.logs['execution'].info('Starting the training procedure for the object detection model...')

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

        # Prepare a test dataset from the validation set taking its first 10 values
        test_path_list = [ann[0] for ann in x_val[:10]]
        test_ds = tf.data.Dataset.from_tensor_slices(test_path_list) \
            .map(self.__resize_fn,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(1) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        # Perform the prediction on the newly created dataset
        detected_predictions = model_utils.predict(model, self.logs['test'], test_ds, steps=10)

        return train_list, get_bb_boxes(detected_predictions, x_val[:10], print=False)

    def run_classification(self, model_params, train_list, bbox_predictions, weights_path):
        """
        Classifies each character according to the available classes via a CNN

        :param model_params: the parameters related to the network
        :param train_list: a train data list predicted at the object detection step
        :param bbox_predictions: the bbox data predicted at the object detection step
        :param weights_path: the path to the saved weights (if present)
        :return: a couple of list with train and bbox data.
        """

        # Generate a model
        model_utils = ModelUtilities()
        model = model_utils.generate_model(input_shape=self.input_shape,
                                           mode=3)

        # Restore the weights, if required
        if model_params['restore_weights']:
            model_utils.restore_weights(model=model,
                                        logger=self.logs['execution'],
                                        init_epoch=model_params['initial_epoch'],
                                        weights_folder_path=weights_path)

        # Compile the model
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(lr=model_params['learning_rate']),
                      metrics=["accuracy"])

        # Generate training set for model 3

        crop_char_path = os.path.join(os.getcwd(), 'datasets', 'char_cropped')

        # NOTE: the following 2 scripts are only run once to generate the images for training.

        # self.logs['execution'].info('Getting bounding boxes from annotations...')
        # crop_format = annotations_to_bounding_boxes(train_list)
        #
        # self.logs['execution'].info('Cropping images to characters...')
        # train_list = create_crop_characters_train(crop_format, crop_char_path)
        # self.logs['execution'].info('Cropping done successfully!')

        # train_list is a list[(image_path, char_class)]
        train_list = load_crop_characters(crop_char_path, mode='train')

        # TODO: now create dataset from cropped images (as [image, category])
        # FIXME: below part is not yet completed

        batch_size = int(model_params['batch_size'])
        self.dataset_params['batch_size'] = batch_size
        dataset_classification = ClassifierDataset(self.dataset_params)
        x_train, x_val = dataset_classification.generate_dataset(train_list)
        classification_ts, classification_ts_size = dataset_classification.get_training_set()
        classification_vs, classification_vs_size = dataset_classification.get_validation_set()

        if model_params['train']:
            self.logs['execution'].info('Starting the training procedure for the classification model...')

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
