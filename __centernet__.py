import logging
import os

import absl.logging
import tensorflow as tf
from typing import List

from networks.classes.SizePredictDataset import SizePredictDataset
from networks.classes.CenterNetDataset import CenterNetDataset
from tensorflow.python.keras.optimizers import Adam, SGD
from networks.classes.ModelUtilities import ModelUtilities as model_utils
from networks.classes.Logger import Logger
from networks.classes.Params import Params
from networks.functions import losses


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

    # Load the general parameters from json file
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
    tra_log = log_handler.get_logger('training')
    tes_log = log_handler.get_logger('testing')

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

    # --------------- MODEL 1 ----------------

    exe_log.info('Building the model...')

    model_1 = model_utils.generate_model(input_shape=input_shape, mode=1)
    model_1.compile(loss='mean_squared_error', optimizer=Adam(lr=model_1_params['learning_rate']))

    weights_path_1 = os.path.join(base_experiments_path, run_id + '_1', 'weights')

    if model_1_params['restore_weights']:
        model_utils.restore_weights(model_1, exe_log, model_1_params['initial_epoch'], weights_path_1)

    # Build dataset
    dataset_params['batch_size'] = model_1_params['batch_size']
    dataset_avg_size = SizePredictDataset(dataset_params)

    dataset_avg_size.generate_dataset()

    sizecheck_ts, sizecheck_ts_size = dataset_avg_size.get_training_set()
    sizecheck_vs, sizecheck_vs_size = dataset_avg_size.get_validation_set()
    # sizecheck_ps, sizecheck_ps_size = dataset_avg_size.get_test_set()

    if model_1_params['train']:
        # Train the model
        exe_log.info('Starting the training procedure for model 1...')

        callbacks = model_utils.setup_callbacks(weights_log_path=weights_path_1,
                                                batch_size=model_1_params['batch_size'])

        model_utils.train(model_1, tra_log, model_1_params['initial_epoch'], model_1_params['epochs'],
                          training_set=sizecheck_ts,
                          validation_set=sizecheck_vs,
                          training_steps=int(sizecheck_ts_size // model_1_params['batch_size'] + 1),
                          validation_steps=int(sizecheck_vs_size // model_1_params['batch_size'] + 1),
                          callbacks=callbacks)

        model_utils.evaluate(model_1, logger=tes_log,
                             evaluation_set=sizecheck_vs,
                             evaluation_steps=int(sizecheck_vs_size // model_1_params['batch_size'] + 1))

    # --------------- MODEL 2 ----------------

    model_2 = model_utils.generate_model(input_shape=input_shape, mode=2)
    model_2.compile(optimizer=Adam(lr=model_2_params['learning_rate']),
                    loss=losses.all_loss,
                    metrics=[losses.size_loss, losses.heatmap_loss, losses.offset_loss])

    weights_path_2 = os.path.join(base_experiments_path, run_id + '_2', 'weights')

    if model_2_params['restore_weights']:
        model_utils.restore_weights(model_2, exe_log, model_2_params['initial_epoch'], weights_path_2)

    if model_2_params['train']:
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

        # Generate dataset for model 2

        dataset_params['batch_size'] = model_2_params['batch_size']
        dataset_detection = CenterNetDataset(dataset_params)
        dataset_detection.generate_dataset(train_list)

        detection_ts, detection_ts_size = dataset_detection.get_training_set()
        detection_vs, detection_vs_size = dataset_detection.get_validation_set()

        # print(detection_ts_size, detection_vs_size)

        # Train the model

        exe_log.info('Starting the training procedure for model 2...')

        callbacks = model_utils.setup_callbacks(weights_log_path=weights_path_2,
                                                batch_size=model_2_params['batch_size'])

        model_utils.train(model_2, tra_log, model_2_params['initial_epoch'], model_2_params['epochs'],
                          training_set=detection_ts,
                          validation_set=detection_vs,
                          training_steps=int(detection_ts_size // model_2_params['batch_size']) + 1,
                          validation_steps=int(detection_vs_size // model_2_params['batch_size']) + 1,
                          callbacks=callbacks)

        model_utils.evaluate(model_2, logger=tes_log,
                             evaluation_set=detection_vs,
                             evaluation_steps=int(detection_vs_size // model_2_params['batch_size'] + 1))

    # --------------- MODEL 3 ----------------

    model_3 = model_utils.generate_model(input_shape=input_shape, mode=3)

    weights_path_3 = os.path.join(base_experiments_path, run_id + '_3', 'weights')

    if model_3_params['restore_weights']:
        model_utils.restore_weights(model_3, exe_log, model_3_params['initial_epoch'],
                                    weights_path_3)

    # --- TEST ---

    # Evaluate training against the given test set


if __name__ == '__main__':
    main()
