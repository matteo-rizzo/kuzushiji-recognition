import pandas as pd
import tensorflow as tf
import numpy as np
import os
from typing import Dict, List, Union

import natsort
from tensorflow.python.keras.optimizers import Adam

from networks.classes.centernet.datasets.DetectionDataset import DetectionDataset
from networks.classes.centernet.models.ModelCenterNet import ModelCenterNet
from networks.classes.centernet.utils.BBoxesHandler import BBoxesHandler
from networks.classes.centernet.utils.Metrics import Metrics
from networks.classes.centernet.models.ModelGeneratorStandard import ModelGeneratorStandard
from networks.classes.centernet.models.ModelGeneratorTile import ModelGeneratorTile


class Detector:

    def __init__(self, model_params, dataset_params, weights_path, logs):
        self.__logs = logs
        self.__weights_path = weights_path

        self.__model_params = model_params
        self.__model_params.update(dataset_params)

        self.__metrics = Metrics()
        self.__bb_handler = BBoxesHandler()
        self.__model_utils = ModelCenterNet(logs=self.__logs)

        self.__model = self.__build_and_compile_model()

        test_list = pd.read_csv(dataset_params['test_csv_path'])['image_id'].to_list()
        base_path = dataset_params['test_images_path']
        self.__test_list = natsort.natsorted([str(os.path.join(base_path, img_id + '.jpg')) for img_id in test_list])

    @staticmethod
    def __resize_fn(path: str, input_h, input_w):
        """
        Utility function for image resizing

        :param path: the path to the image to be resized
        :return: a resized image
        """

        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (input_h, input_w))

        return image_resized / 255

    def __build_and_compile_model(self):

        model_generator = {
            'tile': ModelGeneratorTile(),
            'standard': ModelGeneratorStandard()
        }

        # Generate a model
        model = self.__model_utils.build_model(model_generator=model_generator[self.__model_params['model']],
                                               input_shape=(self.__model_params['input_width'],
                                                            self.__model_params['input_height'],
                                                            self.__model_params['input_channels']),
                                               mode='detection',
                                               n_category=1)

        # Restore the saved weights
        if self.__model_params['restore_weights']:
            self.__model_utils.restore_weights(model=model,
                                               init_epoch=self.__model_params['initial_epoch'],
                                               weights_folder_path=self.__weights_path)

        # Initialize the decay, if defined
        try:
            decay = float(self.__model_params['decay'])
        except ValueError:
            decay = None

        # Compile the model
        model.compile(optimizer=Adam(lr=self.__model_params['learning_rate'],
                                     decay=decay if decay else 0.0),
                      loss=self.__metrics.all_loss,
                      metrics=[self.__metrics.size_loss,
                               self.__metrics.heatmap_loss,
                               self.__metrics.offset_loss])

        return model

    def __train_model(self, dataset):
        self.__logs['execution'].info('Starting the training procedure for the object detection model...')

        # Set up the callbacks
        callbacks = self.__model_utils.setup_callbacks(weights_log_path=self.__weights_path,
                                                       batch_size=self.__model_params['batch_size'],
                                                       lr=self.__model_params['learning_rate'])

        # Start the training procedure
        self.__model_utils.train(dataset=dataset,
                                 model=self.__model,
                                 init_epoch=self.__model_params['initial_epoch'],
                                 epochs=self.__model_params['epochs'],
                                 batch_size=self.__model_params['batch_size'],
                                 callbacks=callbacks)

    def __show_tile_predictions(self, xy_eval):
        self.__logs['execution'].info('Showing examples of tile predictions...')
        _, avg_iou = self.__bb_handler.get_train_tiled_bboxes(xy_eval[:10],
                                                              model=self.__model,
                                                              n_tiles=3,
                                                              show=True)
        self.__logs['execution'].info('The average IoU score using standard model is: {}'.format(avg_iou))

    def __show_standard_predictions(self, xy_eval):
        self.__logs['execution'].info('Showing examples of standard predictions...')

        input_h, input_w = self.__model_params['input_height'], self.__model_params['input_width']

        # Prepare a test dataset from the evaluation set taking its first 10 values
        test_path_list = [ann[0] for ann in xy_eval[:10]]
        mini_test = tf.data.Dataset.from_tensor_slices(test_path_list) \
            .map(lambda i: self.__resize_fn(i, input_h, input_w),
                 num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(1) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        # Perform the prediction on the newly created dataset and show images
        _, avg_iou = self.__bb_handler.get_train_standard_bboxes(self.__model_utils.predict(self.__model, mini_test),
                                                                 annotation_list=xy_eval[:10],
                                                                 show=True)
        self.__logs['execution'].info('The average IoU score using standard model is: {}'.format(avg_iou))

    def __evaluate_model(self, dataset, xy_eval):

        self.__logs['execution'].info('Evaluating the model...')

        evaluation_set, evaluation_set_size = dataset.get_evaluation_set()
        evaluation_steps = evaluation_set_size // self.__model_params['batch_size'] + 1

        metrics = self.__model_utils.evaluate(model=self.__model,
                                              evaluation_set=evaluation_set,
                                              evaluation_steps=evaluation_steps)

        self.__logs['test'].info('Evaluation metrics:\n'
                                 'all_loss     : {}\n'
                                 'size_loss    : {}\n'
                                 'heatmap_loss : {}\n'
                                 'offset_loss  : {}'
                                 .format(metrics[0],
                                         metrics[1],
                                         metrics[2],
                                         metrics[3]))

        # Show some examples of bounding boxes and heatmaps predictions
        if self.__model_params['show_prediction_examples']:
            show_predictions = {
                'tile': self.__show_tile_predictions,
                'standard': self.__show_standard_predictions
            }
            show_predictions[self.__model_params['model']](xy_eval)

    def __generate_tile_predictions(self, test_set=None) -> Dict[str, np.array]:

        self.__logs['execution'].info('Converting test predictions into bounding boxes...')
        return self.__bb_handler.get_test_tiled_bboxes(self.__test_list,
                                                       model=self.__model,
                                                       n_tiles=3,
                                                       show=False)

    def __generate_standard_predictions(self, test_set) -> Dict[str, np.array]:

        self.__logs['execution'].info('Predicting test bounding boxes (takes time)...')
        test_predictions = self.__model_utils.predict(model=self.__model, dataset=test_set)
        self.__logs['execution'].info('Predictions completed.')

        self.__logs['execution'].info('Converting test predictions into bounding boxes...')
        return self.__bb_handler.get_test_standard_bboxes(test_predictions,
                                                          test_images_path=self.__test_list,
                                                          show=True)

    def __generate_test_predictions(self, dataset) -> Dict[str, np.array]:

        test_set, _ = dataset.get_test_set()

        generate_predictions = {
            'tile': self.__generate_tile_predictions,
            'standard': self.__generate_standard_predictions
        }

        predictions = generate_predictions[self.__model_params['model']](test_set)
        self.__logs['execution'].info('Conversion completed.')

        return predictions

    def detect(self, preprocessed_dataset) -> (List[List], Union[Dict[str, np.ndarray], None]):
        """
        Creates and runs a CenterNet to perform the image detection

        :param preprocessed_dataset: a ratio predictor
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

        The returned bbox data consists of a dict with the following structure:
         {<image_path>: np.array[<score>, <ymin>, <xmin>, <ymax>, <xmax>]}
        """

        train_list: List[List] = preprocessed_dataset.get_recommended_splits()

        # Pass the list of test images if we are in test mode,
        # otherwise pass None, so that the test set will not be generated
        self.__test_list = self.__test_list if self.__model_params['predict_on_test'] else None

        # Generate the dataset for detection
        dataset = DetectionDataset(self.__model_params)
        _, _, xy_eval = dataset.generate_dataset(train_list[:10], self.__test_list)

        # Train the model
        if self.__model_params['train']:
            self.__train_model(dataset)

        # Evaluate the model
        if self.__model_params['evaluate']:
            self.__evaluate_model(dataset, xy_eval)

        # Generate the test predictions
        predicted_test_bboxes: Union[Dict[str, np.ndarray], None] = None
        if self.__model_params['predict_on_test']:
            predicted_test_bboxes = self.__generate_test_predictions(dataset)

        return train_list, predicted_test_bboxes
