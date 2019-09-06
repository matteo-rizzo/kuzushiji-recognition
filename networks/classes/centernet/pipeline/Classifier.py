import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Union, Generator

from tensorflow.python.keras.optimizers import Adam

from networks.classes.centernet.datasets.ClassificationDataset import ClassificationDataset
from networks.classes.centernet.models.ModelCenterNet import ModelCenterNet
from networks.classes.centernet.utils.ImageCropper import ImageCropper


class Classifier:

    def __init__(self, model_params, dataset_params, num_categories, weights_path, logs):
        self.__logs = logs
        self.__weights_path = weights_path

        self.__model_params = model_params
        self.__model_params.update(dataset_params)

        self.__model_utils = ModelCenterNet(logs=self.__logs)
        self.__img_cropper = ImageCropper(log=self.__logs['execution'])

        self.__model = self.__build_and_compile_model(num_categories)

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

    def __build_and_compile_model(self, num_categories):
        # Generate a model
        model = self.__model_utils.build_model(input_shape=(self.__model_params['input_width'],
                                                            self.__model_params['input_height'],
                                                            self.__model_params['input_channels']),
                                               mode='classification',
                                               n_category=num_categories)

        # Restore the weights, if required
        if self.__model_params['restore_weights']:
            self.__model_utils.restore_weights(model=model,
                                               init_epoch=self.__model_params['initial_epoch'],
                                               weights_folder_path=self.__weights_path)

        # Initialize the decay, if present
        try:
            decay = float(self.__model_params['decay'])
        except ValueError:
            decay = None

        # Compile the model
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=Adam(lr=self.__model_params['learning_rate'],
                                     decay=decay if decay else 0.0),
                      metrics=["sparse_categorical_accuracy"])

        return model

    def __train_model(self, dataset_classification):

        self.__logs['execution'].info('Starting the training procedure for the classification model...')

        classification_ts, classification_ts_size = dataset_classification.get_training_set()
        classification_vs, classification_vs_size = dataset_classification.get_validation_set()

        callbacks = self.__model_utils.setup_callbacks(weights_log_path=self.__weights_path,
                                                       batch_size=int(self.__model_params['batch_size']),
                                                       lr=self.__model_params['learning_rate'])

        self.__model_utils.train(model=self.__model,
                                 init_epoch=self.__model_params['initial_epoch'],
                                 epochs=self.__model_params['epochs'],
                                 training_set=classification_ts,
                                 validation_set=classification_vs,
                                 training_steps=classification_ts_size // self.__model_params[
                                     'batch_size'] + 1,
                                 validation_steps=classification_vs_size // self.__model_params[
                                     'batch_size'] + 1,
                                 callbacks=callbacks)

    def __evaluate_model(self, dataset_classification):

        self.__logs['execution'].info('Evaluating the classification model...')

        classification_es, classification_es_size = dataset_classification.get_evaluation_set()

        metrics = self.__model_utils.evaluate(model=self.__model,
                                              evaluation_set=classification_es,
                                              evaluation_steps=classification_es_size //
                                                               self.__model_params[
                                                                   'batch_size'] + 1)

        self.__logs['test'].info('Evaluation metrics:\n'
                                 'sparse_categorical_crossentropy : {}\n'
                                 'sparse_categorical_accuracy     : {}'
                                 .format(metrics[0], metrics[1]))

    def __generate_predictions(self, test_list) -> Generator:

        self.__logs['execution'].info(
            'Starting the predict procedure of char class (takes much time)...')

        input_h, input_w = self.__model_params['input_height'], self.__model_params['input_width']

        for img_path in test_list:
            img_dataset = tf.data.Dataset.from_tensor_slices([img_path]) \
                .map(lambda i: self.__resize_fn(i, input_h, input_w),
                     num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                .batch(1)

            prediction = self.__model_utils.predict(model=self.__model, dataset=img_dataset, verbose=0)

            yield prediction

        self.__logs['execution'].info('Prediction completed.')

    def classify(self,
                 train_list: List[List],
                 bbox_predictions: Union[Dict[str, np.ndarray], None]) -> Union[Generator, None]:
        """
        Classifies each character according to the available classes via a CNN

        :param train_list: a train data list predicted at the object detection step
        :param bbox_predictions: the bbox data predicted at the object detection step or None if
                                predictions were not done. dict as {path: score, xmin, ymin, xmax, ymax}
        :return: a couple of list with train and bbox data.
        """

        # Train mode cropping
        train_list = self.__img_cropper.get_crops(img_data=train_list,
                                                  crop_char_path=os.path.join('datasets',
                                                                              'char_cropped_train'),
                                                  regenerate=self.__model_params[
                                                      'regenerate_crops_train'],
                                                  mode='train')

        # Test mode cropping
        test_list: Union[List[str], None] = None
        if self.__model_params['predict_on_test']:
            test_list = self.__img_cropper.get_crops(img_data=bbox_predictions,
                                                     crop_char_path=os.path.join('datasets',
                                                                                 'char_cropped_test'),
                                                     regenerate=self.__model_params[
                                                         'regenerate_crops_test'],
                                                     mode='test')

            self.__write_test_list_to_csv(test_list, bbox_predictions)

        dataset_classification = ClassificationDataset(self.__model_params)
        _, _, xy_eval = dataset_classification.generate_dataset(train_list)

        # Train the model
        if self.__model_params['train']:
            self.__train_model(dataset_classification)

        # Evaluate the model
        if self.__model_params['evaluate']:
            self.__evaluate_model(dataset_classification)

        # Generate predictions
        if self.__model_params['predict_on_test']:
            return self.__generate_predictions(test_list)

        return None

    def __write_test_list_to_csv(self, test_list: List, bbox_predictions: Dict[str, np.array]):

        self.__logs['execution'].info('Writing test data to csv...')

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

            # Set the relative path to the original image
            original_img_path = os.path.join(self.__model_params['test_images_path'],
                                             original_img_name + '.jpg')

            # Convert the coords of the bboxes from float to string
            bbox_coords = [str(coord) for coord in
                           bbox_predictions[original_img_path][cropped_img_id][1:]]

            # Join the coordinates in a single string, in format ymin:xmin:ymax:xmax
            cropped_img_to_bbox[cropped_img_name] = ':'.join(bbox_coords)

        for original_img_name, cropped_img_names in original_img_to_cropped.items():
            for cropped_img_name in cropped_img_names:
                original_img_to_bbox[original_img_name].append(cropped_img_to_bbox[cropped_img_name])

        original_img_to_bbox = {img_name: ' '.join(coords) for img_name, coords in
                                original_img_to_bbox.items()}

        test_list_df = pd.DataFrame({
            'original_image': [original_img_name for original_img_name in
                               original_img_to_cropped.keys()],
            'cropped_images': [' '.join(cropped_img_names) for cropped_img_names in
                               original_img_to_cropped.values()],
            'bboxes': [bbox for bbox in original_img_to_bbox.values()]
        })

        path_to_test_list = os.path.join('datasets', 'test_list.csv')
        test_list_df.to_csv(path_to_test_list)

        self.__logs['execution'].info('Written test data at {}'.format(path_to_test_list))
