import logging
import os

import absl.logging
import tensorflow as tf

from networks.classes.Logger import Logger
from networks.classes.Params import Params
from networks.functions.classification import run_classification
from networks.functions.detection import run_detection
from networks.functions.preprocessing import run_preprocessing


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

    # Set the path to experiments folder
    base_experiments_path = os.path.join(os.getcwd(), 'networks', 'experiments')

    # Load the model parameters from json file
    centernet_params = Params(os.path.join(config_path, 'params_model_CenterNet.json'))
    dataset_params = centernet_params.dataset

    # Get the shape of the input images
    input_shape = (dataset_params['input_width'], dataset_params['input_height'], 3)

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
