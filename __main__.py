import logging
import os

import absl.logging
import tensorflow as tf

from networks.classes.Dataset import Dataset
from networks.classes.Logger import Logger
from networks.classes.ModelCustom import ModelCustom
from networks.classes.ModelYOLO import ModelYOLO
from networks.classes.Params import Params


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

    # Load the general parameters from json file
    general_params = Params(os.path.join(config_path, 'general_params.json'))

    # Get the info for the current run
    run_info = general_params.run_info
    run_id = run_info['id']
    model_name = run_info['model']
    train = run_info['train']
    test = run_info['test']

    # Get the paths to the datasets
    dataset_paths = general_params.datasets

    # Get the ratios of the training, validation and test set
    ratios = general_params.ratios

    # --- LOGGER ---

    log_handler = Logger(run_id)
    log = log_handler.get_logger('execution')

    # Log configuration
    log.info('Software versions:')
    log.info('* Tensorflow version: ' + tf.__version__)
    log.info('* Keras version:      ' + tf.__version__)
    log.info('* Executing eagerly?  ' + eager_exec_status)

    log.info('General parameters:')
    log.info('* Model:              ' + model_name)
    log.info('* Training dataset:   ' + dataset_paths['training'])
    log.info('* Test dataset:       ' + dataset_paths['test'] + '\n')

    # Log general and training parameters
    log_handler.log_configuration(run_id, model_name, implementation=False)

    # --- DATASET ---

    # Import the dataset for training
    log.info('Importing the dataset for training...')
    training_dataset = Dataset(dataset_paths['training'])
    log.info('Training dataset size is: {}'.format(training_dataset.get_size()))

    log.info('Shuffling the dataset...')
    training_dataset.shuffle(3)

    # Import the dataset for testing
    log.info('Importing the dataset for testing...')
    testing_dataset = Dataset(dataset_paths['test'])
    log.info('Testing dataset size is: {}'.format(testing_dataset.get_size()))

    log.info('Shuffling the dataset...')
    testing_dataset.shuffle(3)

    # --- MODEL ---

    log.info('Building the model...')

    # Load the training parameters from json file
    model_params = Params(os.path.join(config_path, 'params_model_' + model_name + '.json'))

    # Set the model
    models = {
        'Custom': ModelCustom,
        'YOLO': ModelYOLO
    }

    # Build the model
    training_set, validation_set, test_set = training_dataset.split()
    model = models[model_name](run_id,
                               model_params,
                               ratios,
                               training_set,
                               validation_set,
                               test_set,
                               log_handler)

    # --- TRAINING ---

    # Train the model
    if train:
        log.info('Starting the training procedure...')
        model.train()

        log.info('Testing the model against an image in the training set...')
        model.predict()

    # --- TEST ---

    # Evaluate training against the given test set
    if test:
        log.info('Evaluating the model...')
        metrics = model.evaluate()
        log_handler.log_metrics(metrics, general_params)


if __name__ == '__main__':
    main()
