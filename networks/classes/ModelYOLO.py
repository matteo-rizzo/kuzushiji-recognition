import os
import cv2
import tensorflow as tf
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

from networks.classes.Logger import Logger
from networks.classes.Model import Model
from libs.darkflow.darkflow.net.build import TFNet


class ModelYOLO(Model):
    def __init__(self,
                 run_id: str,
                 model_params: {},
                 ratios: {},
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 log_handler: Logger):
        # Construct the super class
        super().__init__(run_id,
                         model_params,
                         ratios,
                         training_set,
                         validation_set,
                         test_set,
                         log_handler)

        # Build the YOLO model
        self._build()

    def _build(self):
        """
        Builds the YOLO network.
        """
        # Build the network with the given parameters
        self._model = TFNet(self._network_params)

        img_path = os.path.join(os.getcwd(), 'datasets', 'sample_img', 'sample_multiple_objects.jpg')
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        results = self._model.return_predict(original_img)
        pp.pprint(results)

        plt.subplots(figsize=(10, 10))
        plt.imshow(original_img)
        plt.show()

        plt.subplots(figsize=(20, 10))
        plt.imshow(self.boxing(original_img, results))
        plt.show()

    @staticmethod
    def boxing(original_img, predictions):
        boxed_image = np.copy(original_img)

        for result in predictions:
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))

            if confidence > 0.3:
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
