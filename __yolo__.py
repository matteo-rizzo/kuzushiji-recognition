import logging
import os

import absl.logging
import tensorflow as tf

from networks.classes.general_utilities.Logger import Logger
from networks.classes.yolo.ModelYOLO import ModelYOLO
from networks.classes.general_utilities.Params import Params


def main():
    # -- TENSORFLOW BASIC CONFIG ---

    # Enable eager execution
    # tf.compat.v1.enable_eager_execution()
    eager_exec_status = str('Yes') if tf.compat.v1.executing_eagerly else str('No')

    # Set up the log for tensorflow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Remove absl logs
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

    # --- GENERAL PARAMETERS ---

    # Set the path to the configuration folder
    config_path = os.path.join(os.getcwd(), 'networks', 'configuration')

    # Load the training parameters from json file
    model_params = Params(os.path.join(config_path, 'params_model_YOLO.json'))

    # Get the info for the current run
    run_id = model_params.run_id

    # --- LOGGER ---

    log_handler = Logger(run_id)
    log = log_handler.get_logger('execution')

    # Log configuration
    log.info('Software versions:')
    log.info('* Tensorflow version: ' + tf.__version__)
    log.info('* Keras version:      ' + tf.__version__)
    log.info('* Executing eagerly?  ' + eager_exec_status)

    log.info('General parameters:')
    log.info('* Model:              YOLO')
    log.info('* Training dataset:   ' + model_params.dataset + '\n')

    # Log general and training parameters
    log_handler.log_configuration(run_id, 'YOLO', implementation=False)

    # --- MODEL ---

    log.info('Building the model...')

    # Build the model
    model = ModelYOLO(run_id, model_params, log_handler)

    # --- TRAINING ---

    # Train the model
    if model_params.train:
        log.info('Starting the training procedure...')
        model.train()

        log.info('Testing the model against an image in the training set...')
        model.predict()


if __name__ == '__main__':
    main()
