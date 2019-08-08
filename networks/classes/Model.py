import os
import tensorflow as tf

from networks.classes.Logger import Logger


class Model:
    def __init__(self,
                 run_id: str,
                 model_params: {},
                 ratios: {},
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 log_handler: Logger):
        """
        Builds a model and sets up the train and test sets
        :param run_id: the identification string of the current execution
        :param model_params: the parameters for the selected model
        :param ratios: the ratio of elements to be used for training, validation and test
        :param training_set: the train images
        :param training_set: the validation images
        :param test_set: the images to be used for prediction
        """

        # Set up the datasets
        self._training_set = training_set
        self._validation_set = validation_set
        self._test_set = test_set

        # Set up the parameters for the usage of the model
        self._ratios = ratios
        self._epochs_params = model_params.epochs
        self._network_params = model_params.network

        # Set a flag to state if the model has already been trained
        self._trained = False

        # Set the path to the current experiment
        self._current_experiment_path = os.path.join(os.getcwd(), 'networks', 'experiments', run_id)

        # Initialize an empty model
        self._model = None

        # Initialize the logs
        self._train_log = log_handler.get_logger('training')
        self._test_log = log_handler.get_logger('testing')

    def get_model(self) -> any:
        return self._model

    def _build(self):
        pass

    def _restore_weights(self, experiment_path):
        pass

    def _compile_model(self):
        pass

    def _setup_callbacks(self):
        """
        Sets up the callbacks for the training of the model.
        """
        pass

    def train(self):
        """
        Compiles and trains the model for the specified number of epochs.
        """
        pass

    def plot_model(self):
        """
        Plots the model in png format.
        """
        pass

    def display_summary(self):
        """
        Displays the architecture of the model.
        """
        pass

    def evaluate(self) -> any:
        """
        Evaluates the model returning some key performance indicators.
        """
        pass

    def predict(self):
        """
        Performs a prediction using the model.
        """
        pass
