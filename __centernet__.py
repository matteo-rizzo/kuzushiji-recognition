import logging
import os
import absl.logging
import tensorflow as tf

from networks.classes.centernet.pipeline.Pipeline import CenterNetPipeline
from networks.classes.general_utilities.Logger import Logger
from networks.classes.general_utilities.Params import Params


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

    # --- Paths ---

    # Set the path to the configuration folder
    config_path = os.path.join(os.getcwd(), 'networks', 'configuration')

    # Set the path to experiments folder
    base_experiments_path = os.path.join(os.getcwd(), 'networks', 'experiments')

    # --- Parameters loading ---

    # Load the model parameters from json file
    centernet_params = Params(os.path.join(config_path, 'params_model_CenterNet.json'))
    dataset_params = centernet_params.dataset

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
    exe_log.info('* Model:              CenterNet')
    exe_log.info('* Training dataset:   ' + dataset_params['train_images_path'])
    exe_log.info('* Test dataset:       ' + dataset_params['test_images_path'] + '\n')

    # Log general and training parameters
    log_handler.log_configuration(run_id, 'CenterNet', implementation=False)

    # --- LEARNING PIPELINE --

    # Initialize a learning pipeline for CenterNet
    pipeline = CenterNetPipeline(dataset_params=dataset_params,
                                 logs=logs)

    common_operations = {
        'test_bboxes': ['visualization'],
        'test_submission': ['submission', 'test_submission'],
        'preprocess': ['preprocessing'],
        'detect': ['preprocessing', 'detection'],
        'classify': ['preprocessing', 'detection', 'classification'],
        'write_submission': ['preprocessing', 'detection', 'classification', 'submission'],
        'all': ['preprocessing', 'detection', 'classification', 'submission', 'visualization']
    }

    # Run the pipeline
    pipeline.run_pipeline(operations=common_operations['test_submission'],
                          params=centernet_params,
                          experiment_path=os.path.join(base_experiments_path, run_id))


if __name__ == '__main__':
    main()
