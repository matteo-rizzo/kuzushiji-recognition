import logging
import os
from typing import List

import absl.logging
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from networks.classes.CenterNetDataset import CenterNetDataset
from networks.classes.Logger import Logger
from networks.classes.ModelUtilities import ModelUtilities
from networks.classes.Params import Params
from networks.classes.SizePredictDataset import SizePredictDataset
from networks.functions import losses
from networks.functions.utils import get_bb_boxes


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

    # --- GENERAL PARAMETERS ---

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

    # --- DATASET ---

    # Import the dataset for training
    exe_log.info('Importing the dataset for training...')

    input_shape = (dataset_params['input_width'], dataset_params['input_height'], 3)

    base_experiments_path = os.path.join(os.getcwd(), 'networks', 'experiments')

    # STEP 1: Pre-processing (Check Object Size)

    # exe_log.info('Building the model...')

    # Build dataset for model 1
    dataset_params['batch_size'] = model_1_params['batch_size']
    dataset_avg_size = SizePredictDataset(dataset_params)

    dataset_avg_size.generate_dataset()

    sizecheck_ts, sizecheck_ts_size = dataset_avg_size.get_training_set()
    sizecheck_vs, sizecheck_vs_size = dataset_avg_size.get_validation_set()
    # sizecheck_ps, sizecheck_ps_size = dataset_avg_size.get_test_set()

    # model_1 = model_utils.generate_model(input_shape=input_shape, mode=1)
    # model_1.compile(loss='mean_squared_error', optimizer=Adam(lr=model_1_params['learning_rate']))
    #
    # weights_path_1 = os.path.join(base_experiments_path, run_id + '_1', 'weights')
    #
    # if model_1_params['restore_weights']:
    #     model_utils.restore_weights(model_1, exe_log, model_1_params['initial_epoch'], weights_path_1)
    #
    #
    # if model_1_params['train']:
    #     # Train the model
    #     exe_log.info('Starting the training procedure for model 1...')
    #
    #     callbacks = model_utils.setup_callbacks(weights_log_path=weights_path_1,
    #                                             batch_size=model_1_params['batch_size'])
    #
    #     model_utils.train(model_1, train_log, model_1_params['initial_epoch'], model_1_params['epochs'],
    #                       training_set=sizecheck_ts,
    #                       validation_set=sizecheck_vs,
    #                       training_steps=int(sizecheck_ts_size // model_1_params['batch_size'] + 1),
    #                       validation_steps=int(sizecheck_vs_size // model_1_params['batch_size'] + 1),
    #                       callbacks=callbacks)
    #
    #     model_utils.evaluate(model_1, logger=test_log,
    #                          evaluation_set=sizecheck_vs,
    #                          evaluation_steps=int(sizecheck_vs_size // model_1_params['batch_size'] + 1))

    # STEP 2: Detection by CenterNet

    # Build the CenterNet model
    model_utils = ModelUtilities()
    model_2 = model_utils.generate_model(input_shape=input_shape, mode=2)
    model_2.compile(optimizer=Adam(lr=model_2_params['learning_rate']),
                    loss=losses.all_loss,
                    metrics=[losses.size_loss,
                             losses.heatmap_loss,
                             losses.offset_loss])

    # Set up the path to the weights
    weights_path_2 = os.path.join(base_experiments_path, run_id + '_2', 'weights')

    # Restore the saved weights if required
    if model_2_params['restore_weights']:
        model_utils.restore_weights(model_2, exe_log, model_2_params['initial_epoch'], weights_path_2)

    # Get labels from dataset and compute the recommended split
    avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()

    # flat_predictions = [item for array in predictions for item in array]
    train_list = dataset_avg_size.annotate_split_recommend(avg_sizes)
    # train_list[0] = path to image
    # train_list[1] = annotations (ann)
    # train_list[2] = recommended height split
    # train_list[3] = recommended width split
    # where annotations is data on bbox:
    # ann[:, 1] = xmin
    # ann[:, 2] = ymin
    # ann[:, 3] = x width
    # ann[:, 4] = y height

    # Generate the dataset for model 2

    dataset_params['batch_size'] = model_2_params['batch_size']
    dataset_detection = CenterNetDataset(dataset_params)
    x_train, x_val = dataset_detection.generate_dataset(train_list)
    detection_ts, detection_ts_size = dataset_detection.get_training_set()
    detection_vs, detection_vs_size = dataset_detection.get_validation_set()

    # Train the model
    if model_2_params['train']:
        exe_log.info('Starting the training procedure for model 2 (CenterNet)...')

        callbacks = model_utils.setup_callbacks(weights_log_path=weights_path_2,
                                                batch_size=model_2_params['batch_size'])

        model_utils.train(model=model_2,
                          logger=train_log,
                          init_epoch=model_2_params['initial_epoch'],
                          epochs=model_2_params['epochs'],
                          training_set=detection_ts,
                          validation_set=detection_vs,
                          training_steps=int(detection_ts_size // model_2_params['batch_size']) + 1,
                          validation_steps=int(detection_vs_size // model_2_params['batch_size']) + 1,
                          callbacks=callbacks)

        metrics = model_utils.evaluate(model=model_2,
                                       logger=test_log,
                                       evaluation_set=detection_vs,
                                       evaluation_steps=int(
                                           detection_vs_size // model_2_params['batch_size'] + 1))

        test_log.info('Evaluation metrics:\n'
                      'all_loss     : {}\n'
                      'size_loss    : {}\n'
                      'heatmap_loss : {}\n'
                      'offset_loss  : {}'
                      .format(metrics[0],
                              metrics[1],
                              metrics[2],
                              metrics[3]))

        # Prepare a test dataset from val set. I take the first 10 values of validation set

    def resize_fn(path):
        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (input_shape[1], input_shape[0]))

        return image_resized / 255

    test_path_list = [ann[0] for ann in x_val[:10]]
    test_ds = tf.data.Dataset.from_tensor_slices(test_path_list) \
        .map(resize_fn,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(1) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Get predictions of model 2
    detec_predictions = model_utils.predict(model_2, test_log, test_ds, steps=10)
    bbox_predictions = get_bb_boxes(detec_predictions, x_val[:10], print=True)
    # List of [image_path,category,score,ymin,xmin,ymax,xmax]. Non numeric type!!
    # Category is always 0. It is not the character category. It's the center category.

    # STEP 3: Classification

    model_3 = model_utils.generate_model(input_shape=input_shape, mode=3)

    weights_path_3 = os.path.join(base_experiments_path, run_id + '_3', 'weights')

    if model_3_params['restore_weights']:
        model_utils.restore_weights(model_3, exe_log, model_3_params['initial_epoch'], weights_path_3)

    lr = model_3_params['learning_rate']
    model_3.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr), metrics=["accuracy"])

    # Generate training set for model 3
    # TODO: crop character images
    # TODO: create dataset from cropped images (as [image, category])

    # batch_size_3 = int(model_3_params['batch_size'])
    # dataset_params['batch_size'] = batch_size_3
    # dataset_classification = ClassifierDataset(dataset_params)
    # x_train, x_val = dataset_classification.generate_dataset(bbox_predictions)
    # classification_ts, classification_ts_size = dataset_classification.get_training_set()
    # classification_vs, classification_vs_size = dataset_classification.get_validation_set()
    #
    # if model_3_params['train']:
    #     callbacks = model_utils.setup_callbacks(weights_path_3, batch_size=batch_size_3)
    #
    #     model_utils.train(model_3, train_log, init_epoch=model_3_params['initial_epoch'],
    #                       epochs=model_3_params['epochs'],
    #                       training_set=classification_ts,
    #                       validation_set=classification_vs,
    #                       training_steps=int(classification_ts_size // batch_size_3) + 1,
    #                       validation_steps=int(classification_vs_size // batch_size_3) + 1,
    #                       callbacks=callbacks)

    # --- TEST ---

    # Evaluate training against the given test set


if __name__ == '__main__':
    main()
