import logging

import tensorflow as tf

from networks.classes.Dataset import Dataset
from networks.classes.Model2D import Model2D
from utility_functions.utils import init_loggers, load_parameters, log_configuration, log_metrics

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    # --- GENERAL PARAMETERS AND LOG ---

    # Load the general parameters from json file
    general_params = load_parameters('general_params.json')

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

    # Initialize the logger
    init_loggers(run_id)
    log = logging.getLogger('execution')

    # Log configuration
    log.info('Software versions:')
    log.info('* Tensorflow version: ' + tf.__version__)
    log.info('* Keras version:      ' + tf.__version__)

    log.info('General parameters:')
    log.info('* Model:              ' + model_name)
    log.info('* Training dataset:   ' + dataset_paths['training'])
    log.info('* Test dataset:       ' + dataset_paths['test'] + '\n')

    # Log general and training parameters
    log_configuration(run_id, model_name)

    # --- DATASET ---

    # Import the dataset for training
    logging.info('Importing the dataset for training...')
    training_dataset = Dataset(dataset_paths['training'])

    log.info('Shuffling the dataset...')
    training_dataset.shuffle(3)

    # Import the dataset for testing
    logging.info('Importing the dataset for testing...')
    testing_dataset = Dataset(dataset_paths['test'])

    log.info('Shuffling the dataset...')
    testing_dataset.shuffle(3)

    # Split the dataset
    log.info('Splitting the dataset with training size: ' + str(ratios['training']) + '\n')
    training_set, validation_set, test_set = training_dataset.split(ratios['training'], ratios['validation'])

    # --- MODEL ---

    log.info('Building the model...')

    # Load the training parameters from json file
    model_params = load_parameters('params_model' + model_name + '.json')

    # Setting the model
    models = {
        '2D': Model2D
    }

    # Build the model
    model = models[model_name](training_set,
                               validation_set,
                               test_set,
                               model_params)

    # --- TRAINING ---

    # Train the model
    if train:
        log.info('Starting the training procedure...')
        model.train()

    # --- TEST ---

    # Evaluate training against the given test set
    if test:
        log.info('Evaluating the model...')
        metrics = model.evaluate()
        log_metrics(metrics, general_params, 'testing')


if __name__ == '__main__':
    main()
