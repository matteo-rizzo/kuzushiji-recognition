import os
import pprint as pp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from libs.darkflow.darkflow.net.build import TFNet

from networks.classes.general_utilities.Logger import Logger
from networks.classes.yolo.Model import Model


class ModelYOLO(Model):
    def __init__(self,
                 run_id: str,
                 model_params: {},
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 log_handler: Logger):
        """
        Initializes a YOLO network v2.
        :param run_id: the identification name of the current run
        :param model_params: the parameters of the network
        :param ratios: the ratio of elements to be used for training, validation and test
        :param training_set: the training images
        :param validation_set: the validation images
        :param test_set: the test images
        :param log_handler: a logger
        """

        # Construct the super class
        super().__init__(run_id,
                         model_params,
                         training_set,
                         validation_set,
                         test_set,
                         60,
                         30,
                         10,
                         log_handler)

        # Build the YOLO model
        self._build()

    def _build(self):
        """
        Builds the YOLO network.
        """

        self._model = TFNet(self._network_params)

    def train(self):
        """
        Trains the model for the specified number of epochs.
        """

        self._model.train()

    def predict(self):
        """
        Performs a prediction using the model.
        """

        image_file_name = '100241706_00006_1.jpg'
        threshold = self._network_params['threshold']

        img_path = os.path.join(os.getcwd(),
                                'datasets',
                                'kaggle',
                                'training',
                                'images',
                                image_file_name)

        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        results = self._model.return_predict(original_img)
        self._test_log.info('Test image data:')
        self._test_log.info('* Image     :    {}'.format(image_file_name))
        self._test_log.info('* Threshold :    {}\n'.format(threshold))
        self._test_log.info('{} objects detected:\n'.format(len(results)))
        self._test_log.info(pp.pformat(results))
        print(pp.pformat(results))

        plt.subplots(figsize=(10, 10))
        plt.imshow(original_img)
        plt.show()

        plt.subplots(figsize=(10, 10))
        plt.imshow(self.boxing(original_img,
                               results,
                               threshold))
        plt.show()

    @staticmethod
    def boxing(original_img, predictions, threshold):
        boxed_image = np.copy(original_img)

        scale = 1

        for result in predictions:
            top_x = int(result['topleft']['x'] * scale)
            top_y = int(result['topleft']['y'] * scale)

            btm_x = int(result['bottomright']['x'] * scale)
            btm_y = int(result['bottomright']['y'] * scale)

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))

            if confidence >= threshold:
                boxed_image = cv2.rectangle(img=boxed_image,
                                            pt1=(top_x, top_y),
                                            pt2=(btm_x, btm_y),
                                            color=(255, 0, 0),
                                            thickness=3)

                boxed_image = cv2.putText(img=boxed_image,
                                          text=label,
                                          org=(top_x, top_y - 5),
                                          fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                          fontScale=0.8,
                                          color=(0, 230, 0),
                                          thickness=1,
                                          lineType=cv2.LINE_AA)

        return boxed_image
