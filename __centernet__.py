import logging
import os

import absl.logging
import tensorflow as tf

from networks.classes.CenterNetPipeline import CenterNetPipeline
from networks.classes.Logger import Logger
from networks.classes.Params import Params


def main():
    # -- TENSORFLOW BASIC CONFIG ---

    # Enable eager execution
    # tf.compat.v1.enable_eager_execution()
    eager_exec_status = str('Yes') if tf.executing_eagerly() else str('No')

    # Set up the log for tensorflow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Remove absl logs
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

    # --- PARAMETERS INITIALIZATION ---

    # --- Paths ---

    # Set the path to the configuration folder
    config_path = os.path.join(os.getcwd(), 'networks', 'configuration')

    # Set the path to experiments folder
    base_experiments_path = os.path.join(os.getcwd(), 'networks', 'experiments')

    # --- Parameters loading ---

    # Load the model parameters from json file
    centernet_params = Params(os.path.join(config_path, 'params_model_CenterNet.json'))
    dataset_params = centernet_params.dataset

    # --- Main parameters ---

    # Model params
    model_1_params = centernet_params.model_1
    model_2_params = centernet_params.model_2
    model_3_params = centernet_params.model_3

    # Get the info for the current run
    run_id = centernet_params.run_id

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

    # --- LEARNING PIPELINE --

    # Initialize a learning pipeline for CenterNet
    pipeline = CenterNetPipeline(dataset_params=dataset_params,
                                 logs=logs)

    # --- STEP 1: Pre-processing (Check Object Size) ---

    dataset_avg_size = pipeline.run_preprocessing(model_params=model_1_params,
                                                  weights_path=os.path.join(base_experiments_path,
                                                                            run_id + '_1',
                                                                            'weights'))

    # --- STEP 2: Detection by CenterNet ---

    pipeline.run_hourglass_detection(model_params=model_2_params,
                                     dataset_avg_size=dataset_avg_size,
                                     weights_path=os.path.join(
                                         base_experiments_path,
                                         run_id + '_2',
                                         'weights'),
                                     run_id=run_id)

    train_list, bbox_predictions = pipeline.run_detection(model_params=model_2_params,
                                                          dataset_avg_size=dataset_avg_size,
                                                          weights_path=os.path.join(
                                                              base_experiments_path,
                                                              run_id + '_2',
                                                              'weights'))

    # --- STEP 3: Classification ---

    pipeline.run_classification(model_params=model_3_params,
                                train_list=train_list,
                                bbox_predictions=bbox_predictions,
                                weights_path=os.path.join(base_experiments_path, run_id + '_3',
                                                          'weights'))


if __name__ == '__main__':
    main()
