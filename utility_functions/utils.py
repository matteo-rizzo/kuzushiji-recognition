import logging
import os
import sys
from shutil import copy

from networks.classes.Params import Params


def init_loggers(run_id):
    """
    Initialize the loggers.
    """
    # Calculate the current timestamp
    # run_id = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Set a format
    frm = '%(asctime)s - %(levelname)s - %(message)s'

    # Set the path to the experiments folder
    experiments_path = os.path.join(os.getcwd(), "networks", "experiments")
    os.makedirs(experiments_path, exist_ok=True)

    # Set up a new directory for the current experiment
    log_directory_name = run_id
    log_directory_path = os.path.join(experiments_path, log_directory_name)
    os.makedirs(log_directory_path, exist_ok=True)

    # Create a logger for the execution
    exec_log_path = os.path.join(log_directory_path, 'execution.log')
    exec_logger = logging.getLogger('execution')
    exec_logger.setLevel('INFO')

    fh = logging.FileHandler(exec_log_path, mode='w')
    fh.setFormatter(logging.Formatter(frm))
    exec_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(frm))
    exec_logger.addHandler(sh)

    # Create a logger for the training
    train_log_path = os.path.join(log_directory_path, 'training.log')
    train_logger = logging.getLogger('training')
    train_logger.setLevel('INFO')

    fh = logging.FileHandler(train_log_path, mode='a')
    fh.setFormatter(logging.Formatter(frm))
    train_logger.addHandler(fh)

    # Create a logger for the testing
    test_log_path = os.path.join(log_directory_path, 'test.log')
    test_logger = logging.getLogger('testing')
    test_logger.setLevel('INFO')
    fh = logging.FileHandler(test_log_path, mode='a')
    fh.setFormatter(logging.Formatter(frm))
    test_logger.addHandler(fh)


def log_configuration(run_id, model):
    """
    Log the parameters json files for the current experiment creating a copy of them.
    :param run_id: the identification code for the current experiment
    :param model: the selected model (i.e. 2D or 3D)
    """
    # Path to the configuration folder
    config_path = os.path.join(os.getcwd(), 'networks', 'configuration')

    # Path to classes
    class_path = os.path.join(os.getcwd(), 'networks', 'classes')

    # Path to the log of the current experiment
    experiment_path = os.path.join(os.getcwd(), 'networks', 'experiments', run_id)

    # Path to the configuration log for the current experiment
    config_log_path = os.path.join(experiment_path, 'configuration')
    os.makedirs(config_log_path, exist_ok=True)

    # Log general parameters
    copy(os.path.join(config_path, 'general_params.json'), config_log_path)

    # Log model parameters
    copy(os.path.join(config_path, 'params_model' + model + '.json'), config_log_path)

    # Log network architecture
    copy(os.path.join(class_path, 'Model' + model + '.py'), config_log_path)


def log_metrics(metrics, params, log_type):
    """
    Log the loss and accuracy metrics for the current experiment.
    :param log_type: the type of logger to be used (i.e. training, test etc...)
    :param params: general params object
    :param metrics: loss and accuracy for the current experiment
    """
    log = logging.getLogger(log_type)

    log.info('Test set: ' + str(params.test_dataset))
    log.info('Metrics:')
    log.info('* Loss:        ' + str(metrics[0]))
    log.info('* Accuracy:    ' + str(metrics[1]) + '\n')


def load_parameters(params_filename) -> Params:
    """
    Load parameters from json file.
    :param params_filename: the name of the json file containing the parameters
    stored in the configuration folder
    :return: a Params object storing all the desired parameters
    """
    json_path = os.path.join(os.getcwd(), 'networks', 'configuration', params_filename)
    logging.getLogger('execution').info('Loading parameters from {}...\n'.format(json_path))
    assert os.path.isfile(json_path), 'ERR: No json configuration file found at {}'.format(json_path)

    return Params(json_path)


def save_best_weights(weights_log_path, best_weights_path):
    """
    Save the weights of the current experiment to the best_weights folder in experiments.
    :param weights_log_path: the path to the folder which the weights of the current experiment have been logged to
    :param best_weights_path: the path to the best_weights folder in experiments
    """
    for _, _, f in os.walk(weights_log_path):
        for file in f:
            copy(os.path.join(weights_log_path, file), best_weights_path)
