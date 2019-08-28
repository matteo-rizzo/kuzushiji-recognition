import logging
import os
from typing import List

import absl.logging
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from networks.classes.CenterNetDataset import CenterNetDataset
from networks.classes.ClassifierDataset import ClassifierDataset
from networks.classes.Logger import Logger
from networks.classes.ModelUtilities import ModelUtilities
from networks.classes.Params import Params
from networks.classes.SizePredictDataset import SizePredictDataset
from networks.functions import losses
from networks.functions.utils import get_bb_boxes, create_crop_characters_train, \
    annotations_to_bounding_boxes, load_crop_characters


def run_preprocessing(dataset_params, model_params, input_shape, weights_path,
                      logs) -> SizePredictDataset:
    """
    Creates and runs a CNN which takes an image/page of manuscript as input and predicts the
    average dimensional ratio between the characters and the image itself

    :param dataset_params: the parameters related to the dataset
    :param model_params: the parameters related to the network
    :param input_shape: the input shape of the images (usually 512x512x3)
    :param weights_path: the path to the saved weights (if present)
    :param logs: the loggers (execution, training and test)
    :return: a ratio predictor
    """

    logs['execution'].info('Preprocessing the data...')

    # Build dataset for model 1
    dataset_params['batch_size'] = model_params['batch_size']
    dataset_avg_size = SizePredictDataset(dataset_params)

    dataset_avg_size.generate_dataset()

    size_check_ts, size_check_ts_size = dataset_avg_size.get_training_set()
    size_check_vs, size_check_vs_size = dataset_avg_size.get_validation_set()
    # size_check_ps, size_check_ps_size = dataset_avg_size.get_test_set()
    #
    # # Generate a model
    # model_utils = ModelUtilities()
    # model = model_utils.generate_model(input_shape=input_shape, mode=1)
    # model.compile(loss='mean_squared_error',
    #               optimizer=Adam(lr=model_params['learning_rate']))
    #
    # # Restore the weights, if required
    # if model_params['restore_weights']:
    #     model_utils.restore_weights(model,
    #                                 logs['execution'],
    #                                 model_params['initial_epoch'],
    #                                 weights_path)
    #
    # # Train the model
    # if model_params['train']:
    #     logs['execution'].info('Starting the training procedure for model 1...')
    #
    #     # Set up the callbacks
    #     callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
    #                                             batch_size=model_params['batch_size'])
    #
    #     # Start the training procedure
    #     model_utils.train(model, logs['training'], model_params['initial_epoch'], model_params['epochs'],
    #                       training_set=size_check_ts,
    #                       validation_set=size_check_vs,
    #                       training_steps=int(size_check_ts_size // model_params['batch_size'] + 1),
    #                       validation_steps=int(size_check_vs_size // model_params['batch_size'] + 1),
    #                       callbacks=callbacks)
    #
    #     # Evaluate the model
    #     model_utils.evaluate(model, logger=logs['test'],
    #                          evaluation_set=size_check_vs,
    #                          evaluation_steps=int(size_check_vs_size // model_params['batch_size'] + 1))

    return dataset_avg_size


def run_detection(dataset_params,
                  model_params,
                  dataset_avg_size,
                  input_shape,
                  weights_path,
                  logs) -> (List, List):
    """
    Creates and runs a CenterNet to perform the image detection

    :param dataset_params: the parameters related to the dataset
    :param model_params: the parameters related to the network
    :param dataset_avg_size: a ratio predictor
    :param input_shape: the input shape of the images (usually 512x512x3)
    :param weights_path: the path to the saved weights (if present)
    :param logs: the loggers (execution, training and test)
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
    model = model_utils.generate_model(input_shape=input_shape, mode=2)
    model.compile(optimizer=Adam(lr=model_params['learning_rate']),
                  loss=losses.all_loss,
                  metrics=[losses.size_loss,
                           losses.heatmap_loss,
                           losses.offset_loss])

    # Restore the saved weights if required
    if model_params['restore_weights']:
        model_utils.restore_weights(model=model,
                                    logger=logs['execution'],
                                    init_epoch=model_params['initial_epoch'],
                                    weights_folder_path=weights_path)

    # Get labels from dataset and compute the recommended split
    avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
    train_list = dataset_avg_size.annotate_split_recommend(avg_sizes)

    # Generate the dataset for detection
    dataset_params['batch_size'] = model_params['batch_size']
    dataset_detection = CenterNetDataset(dataset_params)
    x_train, x_val = dataset_detection.generate_dataset(train_list)
    detection_ts, detection_ts_size = dataset_detection.get_training_set()
    detection_vs, detection_vs_size = dataset_detection.get_validation_set()

    # Train the model
    if model_params['train']:
        logs['execution'].info('Starting the training procedure for model 2 (CenterNet)...')

        # Set up the callbacks
        callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
                                                batch_size=model_params['batch_size'])

        # Start the training procedure
        model_utils.train(model=model,
                          logger=logs['training'],
                          init_epoch=model_params['initial_epoch'],
                          epochs=model_params['epochs'],
                          training_set=detection_ts,
                          validation_set=detection_vs,
                          training_steps=int(detection_ts_size // model_params['batch_size']) + 1,
                          validation_steps=int(detection_vs_size // model_params['batch_size']) + 1,
                          callbacks=callbacks)

        # Evaluate the model
        metrics = model_utils.evaluate(model=model,
                                       logger=logs['test'],
                                       evaluation_set=detection_vs,
                                       evaluation_steps=int(
                                           detection_vs_size // model_params['batch_size'] + 1))

        logs['test'].info('Evaluation metrics:\n'
                          'all_loss     : {}\n'
                          'size_loss    : {}\n'
                          'heatmap_loss : {}\n'
                          'offset_loss  : {}'
                          .format(metrics[0],
                                  metrics[1],
                                  metrics[2],
                                  metrics[3]))

    # Utility function for resizing
    def resize_fn(path):
        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (input_shape[1], input_shape[0]))

        return image_resized / 255

    # Prepare a test dataset from the validation set taking its first 10 values
    test_path_list = [ann[0] for ann in x_val[:10]]
    test_ds = tf.data.Dataset.from_tensor_slices(test_path_list) \
        .map(resize_fn,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(1) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Perform the prediction on the newly created dataset
    detected_predictions = model_utils.predict(model, logs['test'], test_ds, steps=10)

    # get_bb_boxes returns a dict of {image_path: [category,score,ymin,xmin,ymax,xmax]}.
    # Category is always 0. It is not the character category. It's the center category.
    return train_list, get_bb_boxes(detected_predictions, x_val[:10], print=False)


def run_classification(dataset_params,
                       model_params,
                       train_list,
                       bbox_predictions,
                       input_shape,
                       weights_path,
                       logs):
    """
    Classifies each character according to the available classes via a CNN

    :param dataset_params: the parameters related to the dataset
    :param model_params: the parameters related to the network
    :param train_list: a train data list predicted at the object detection step
    :param bbox_predictions: the bbox data predicted at the object detection step
    :param input_shape: the input shape of the images (usually 512x512x3)
    :param weights_path: the path to the saved weights (if present)
    :param logs: the loggers (execution, training and test)
    :return: a couple of list with train and bbox data.
    """

    # Generate a model
    model_utils = ModelUtilities()
    model = model_utils.generate_model(input_shape=input_shape,
                                       mode=3)

    # Restore the weights, if required
    if model_params['restore_weights']:
        model_utils.restore_weights(model=model,
                                    logger=logs['execution'],
                                    init_epoch=model_params['initial_epoch'],
                                    weights_folder_path=weights_path)

    # Compile the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=model_params['learning_rate']),
                  metrics=["accuracy"])

    # Generate training set for model 3

    crop_char_path = os.path.join(os.getcwd(), 'datasets', 'char_cropped')

    # NOTE: the following 2 scripts are only run once to generate the images for training.

    # logs['execution'].info('Getting bounding boxes from annotations...')
    # crop_format = annotations_to_bounding_boxes(train_list)

    # logs['execution'].info('Cropping images to characters...')
    # char_train_list = create_crop_characters_train(crop_format, crop_char_path)

    # logs['execution'].info('Cropping done successfully!')

    train_list_3 = load_crop_characters(crop_char_path, mode='train')
    # train_list_3 is a list[(image_path, char_class)]

    # TODO: now create dataset from cropped images (as [image, category])
    # FIXME: below part is not yet completed

    batch_size = int(model_params['batch_size'])
    dataset_params['batch_size'] = batch_size
    dataset_classification = ClassifierDataset(dataset_params)
    x_train, x_val = dataset_classification.generate_dataset(train_list_3)
    classification_ts, classification_ts_size = dataset_classification.get_training_set()
    classification_vs, classification_vs_size = dataset_classification.get_validation_set()

    if model_params['train']:
        callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
                                                batch_size=batch_size)

        model_utils.train(model=model,
                          logger=logs['training'],
                          init_epoch=model_params['initial_epoch'],
                          epochs=model_params['epochs'],
                          training_set=classification_ts,
                          validation_set=classification_vs,
                          training_steps=int(classification_ts_size // batch_size) + 1,
                          validation_steps=int(classification_vs_size // batch_size) + 1,
                          callbacks=callbacks)


def main():
    # -- TENSORFLOW BASIC CONFIG ---

    # Enable eager execution
    tf.compat.v1.enable_eager_execution()
    eager_exec_status = str('Yes') if tf.executing_eagerly() else str('No')

    # Set up the log for tensorflow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Remove absl logs
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

    # --- PARAMETERS INITIALIZATION ---

    # Set the path to the configuration folder
    config_path = os.path.join(os.getcwd(), 'networks', 'configuration')

    # Load the model parameters from json file
    centernet_params = Params(os.path.join(config_path, 'params_model_CenterNet.json'))
    dataset_params = centernet_params.dataset

    # Get the info for the current run
    run_id = centernet_params.run_id

    # Model params
    model_1_params = centernet_params.model_1
    model_2_params = centernet_params.model_2
    model_3_params = centernet_params.model_3

    # --- LOGGERS ---

    log_handler = Logger(run_id)

    exe_log = log_handler.get_logger('execution')
    train_log = log_handler.get_logger('training')
    test_log = log_handler.get_logger('testing')

    logs = {
        'execution': exe_log,
        'training': train_log,
        'test': test_log
    }

    # Log configuration
    exe_log.info('Software versions:')
    exe_log.info('* Tensorflow version: ' + tf.__version__)
    exe_log.info('* Keras version:      ' + tf.__version__)
    exe_log.info('* Executing eagerly?  ' + eager_exec_status)

    exe_log.info('General parameters:')
    exe_log.info('* Model:               CenterNet')
    exe_log.info('* Training dataset:   ' + dataset_params['train_images_path'])
    exe_log.info('* Test dataset:       ' + dataset_params['test_images_path'] + '\n')

    # Log general and training parameters
    log_handler.log_configuration(run_id, 'CenterNet', implementation=False)

    input_shape = (dataset_params['input_width'], dataset_params['input_height'], 3)

    base_experiments_path = os.path.join(os.getcwd(), 'networks', 'experiments')

    # --- STEP 1: Pre-processing (Check Object Size) ---

    dataset_avg_size = run_preprocessing(dataset_params=dataset_params,
                                         model_params=model_1_params,
                                         input_shape=input_shape,
                                         weights_path=os.path.join(base_experiments_path, run_id + '_1',
                                                                   'weights'),
                                         logs=logs)

    # --- STEP 2: Detection by CenterNet ---

    train_list, bbox_predictions = run_detection(dataset_params=dataset_params,
                                                 model_params=model_2_params,
                                                 dataset_avg_size=dataset_avg_size,
                                                 input_shape=input_shape,
                                                 weights_path=os.path.join(base_experiments_path,
                                                                           run_id + '_2',
                                                                           'weights'),
                                                 logs=logs)

    # --- STEP 3: Classification ---

    run_classification(dataset_params=dataset_params,
                       model_params=model_3_params,
                       train_list=train_list,
                       bbox_predictions=bbox_predictions,
                       input_shape=input_shape,
                       weights_path=os.path.join(base_experiments_path, run_id + '_3',
                                                 'weights'),
                       logs=logs)


if __name__ == '__main__':
    main()
