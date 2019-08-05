import logging
import os

from shutil import copy


class Logger:

    def __init__(self, run_id: str):
        """
        Initialize the loggers.
        """
        # Set up a formatter
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Set the path to the experiments folder
        experiments_path = os.path.join(os.getcwd(), "networks", "experiments")
        os.makedirs(experiments_path, exist_ok=True)

        # Set up a new directory for the current experiment
        log_directory_name = run_id
        self.log_directory_path = os.path.join(experiments_path, log_directory_name)
        os.makedirs(self.log_directory_path, exist_ok=True)

        # Set up the loggers
        self.exec_logger = self.__set_execution_logger()
        self.train_logger = self.__set_training_logger()
        self.test_logger = self.__set_testing_logger()

    def __set_execution_logger(self) -> logging.Logger:
        """
        Create a logger for the execution
        """
        exec_log_path = os.path.join(self.log_directory_path, 'execution.log')
        exec_logger = logging.getLogger('execution')

        if exec_logger.hasHandlers():
            exec_logger.handlers.clear()

        exec_logger.setLevel('INFO')

        ch = logging.StreamHandler()
        fh = logging.FileHandler(exec_log_path, mode='a')

        ch.setLevel('INFO')
        fh.setLevel('INFO')

        ch.setFormatter(self.formatter)
        fh.setFormatter(self.formatter)

        exec_logger.addHandler(ch)
        exec_logger.addHandler(fh)

        return exec_logger

    def __set_training_logger(self) -> logging.Logger:
        """
        Create a logger for training
        """
        train_log_path = os.path.join(self.log_directory_path, 'training.log')
        train_logger = logging.getLogger('training')
        train_logger.setLevel('INFO')

        fh = logging.FileHandler(train_log_path, mode='a')
        train_logger.addHandler(fh)

        return train_logger

    def __set_testing_logger(self) -> logging.Logger:
        """
        Create a logger for testing
        """
        test_log_path = os.path.join(self.log_directory_path, 'test.log')
        test_logger = logging.getLogger('testing')
        test_logger.setLevel('INFO')

        fh = logging.FileHandler(test_log_path, mode='a')
        test_logger.addHandler(fh)

        return test_logger

    def get_logger(self, log_type: str) -> logging.Logger:

        loggers = {
            'execution': self.exec_logger,
            'training': self.train_logger,
            'testing': self.test_logger
        }

        return loggers[log_type]

    @staticmethod
    def log_configuration(run_id: str, model_name, implementation: bool = False):
        """
        Log the parameters json files for the current experiment creating a copy of them.
        :param run_id: the identification code for the current experiment
        :param model_name: the name of the selected model (i.g. YOLO)
        :param implementation: boolean flag to log the implementation of the model
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
        copy(os.path.join(config_path, 'params_model_' + model_name + '.json'), config_log_path)

        # Log network architecture
        if implementation:
            copy(os.path.join(class_path, 'Model' + model_name + '.py'), config_log_path)

    def log_metrics(self, metrics: (float, float), params: {}):
        """
        Log the loss and accuracy metrics for the current experiment.
        :param metrics: loss and accuracy for the current experiment
        :param params: general params object
        """

        self.test_logger.info('Test set:    ' + str(params.test_dataset))
        self.test_logger.info('Metrics:')
        self.test_logger.info('* Loss:      ' + str(metrics[0]))
        self.test_logger.info('* Accuracy:  ' + str(metrics[1]) + '\n')

    @staticmethod
    def save_best_weights(weights_log_path: str, best_weights_path: str):
        """
        Save the weights of the current experiment to the best_weights folder in experiments.
        :param weights_log_path: the path to the folder which the weights of the current experiment have been logged to
        :param best_weights_path: the path to the best_weights folder in experiments
        """
        for _, _, f in os.walk(weights_log_path):
            for file in f:
                copy(os.path.join(weights_log_path, file), best_weights_path)
