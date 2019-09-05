import pandas as pd
import tensorflow as tf
import numpy as np
import os
from typing import Dict, List, Union
from tensorflow.python.keras.optimizers import Adam

from networks.classes.centernet.datasets.DetectionDataset import DetectionDataset
from networks.classes.centernet.models.ModelCenterNet import ModelCenterNet
from networks.classes.centernet.utils.BBoxesHandler import BBoxesHandler
from networks.classes.centernet.utils.LossFunctionsGenerator import LossFunctionsGenerator


class Detector:

    def __init__(self, model_params, dataset_params, weights_path, logs):
        self.__logs = logs

        self.__model_params = model_params
        self.__model_params.update(dataset_params)

        self.__loss = LossFunctionsGenerator()
        self.__model_utils = ModelCenterNet(logs=self.__logs)
        self.__bb_handler = BBoxesHandler()

        self.__weights_path = weights_path
        self.__model = self.__build_and_compile_model()

        test_list = pd.read_csv(dataset_params['sample_submission'])['image_id'].to_list()
        base_path = dataset_params['test_images_path']
        self.__test_list = [str(os.path.join(base_path, img_id + '.jpg')) for img_id in test_list]

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
        model = self.__model_utils.build_model(input_shape=(self.__model_params['input_width'],
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
                      loss=self.__loss.all_loss,
                      metrics=[self.__loss.size_loss,
                               self.__loss.heatmap_loss,
                               self.__loss.offset_loss])

        return model

    def __train_model(self, dataset_detection):
        self.__logs['execution'].info('Starting the training procedure for the object detection model...')

        detection_ts, detection_ts_size = dataset_detection.get_training_set()
        detection_vs, detection_vs_size = dataset_detection.get_validation_set()

        # Set up the callbacks
        callbacks = self.__model_utils.setup_callbacks(weights_log_path=self.__weights_path,
                                                       batch_size=self.__model_params['batch_size'],
                                                       lr=self.__model_params['learning_rate'])

        # Start the training procedure
        self.__model_utils.train(model=self.__model,
                                 init_epoch=self.__model_params['initial_epoch'],
                                 epochs=self.__model_params['epochs'],
                                 training_set=detection_ts,
                                 validation_set=detection_vs,
                                 training_steps=int(detection_ts_size // self.__model_params['batch_size']) + 1,
                                 validation_steps=int(detection_vs_size // self.__model_params['batch_size']) + 1,
                                 callbacks=callbacks)

    def __evaluate_model(self, dataset_detection, xy_eval):

        self.__logs['test'].info('Evaluating the model...')

        detection_es, detection_es_size = dataset_detection.get_evaluation_set()

        metrics = self.__model_utils.evaluate(model=self.__model,
                                              evaluation_set=detection_es,
                                              evaluation_steps=int(
                                                  detection_es_size // self.__model_params[
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

        # Show some examples of bounding boxes and heatmaps predictions
        if self.__model_params['show_prediction_examples']:
            input_h, input_w = self.__model_params['input_height'], self.__model_params['input_width']
            self.__logs['test'].info('Showing prediction examples...')

            # Prepare a test dataset from the evaluation set taking its first 10 values
            test_path_list = [ann[0] for ann in xy_eval[:10]]
            mini_test = tf.data.Dataset.from_tensor_slices(test_path_list) \
                .map(lambda i: self.__resize_fn(i, input_h, input_w),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .batch(1) \
                .prefetch(tf.data.experimental.AUTOTUNE)

            # Perform the prediction on the newly created dataset and show images
            detected_predictions = self.__model_utils.predict(self.__model, mini_test)
            self.__bb_handler.get_bboxes(detected_predictions,
                                         mode='train',
                                         annotation_list=xy_eval[:10],
                                         show=True)

    def __generate_test_predictions(self, dataset_detection):

        detection_ps, detection_ps_size = dataset_detection.get_test_set()

        self.__logs['execution'].info('Predicting test bounding boxes (takes time)...')
        test_predictions = self.__model_utils.predict(model=self.__model, dataset=detection_ps)
        self.__logs['execution'].info('Predictions completed.')

        self.__logs['execution'].info('Converting test predictions into bounding boxes...')
        predicted_test_bboxes = self.__bb_handler.get_bboxes(test_predictions,
                                                             mode='test',
                                                             test_images_path=self.__test_list,
                                                             show=False)
        self.__logs['execution'].info('Conversion completed.')

        return predicted_test_bboxes

    def detect(self, dataset_avg_size) -> (List[List], Union[Dict[str, np.ndarray], None]):
        """
        Creates and runs a CenterNet to perform the image detection

        :param dataset_avg_size: a ratio predictor
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

        # Get labels from dataset and compute the recommended split,
        # format is: [ [image path, annotations, height split, width split] ]
        avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
        train_list: List[List] = dataset_avg_size.get_recommended_splits(avg_sizes)

        # Pass the list of test images if we are in test mode,
        # otherwise pass None, so that the test set will not be generated
        test_list = self.__test_list if self.__model_params['predict_on_test'] else None

        # Generate the dataset for detection
        dataset_detection = DetectionDataset(self.__model_params)
        xy_train, xy_val, xy_eval = dataset_detection.generate_dataset(train_list, test_list)

        # Train the model
        if self.__model_params['train']:
            self.__train_model(dataset_detection)

        # Evaluate the model
        if self.__model_params['evaluate']:
            self.__evaluate_model(dataset_detection, xy_eval)

        # Generate the test predictions
        predicted_test_bboxes: Union[Dict[str, np.ndarray], None] = None
        if self.__model_params['predict_on_test']:
            predicted_test_bboxes = self.__generate_test_predictions(dataset_detection)

        return train_list, predicted_test_bboxes
