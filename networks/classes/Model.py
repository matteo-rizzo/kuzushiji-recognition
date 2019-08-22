import os
import glob

import tensorflow as tf
from tensorflow.python.keras.utils import plot_model

from networks.classes.Logger import Logger


class Model:
    def __init__(self,
                 run_id: str,
                 model_params: {},
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 train_size: int,
                 val_size: int,
                 test_size: int,
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
        self._train_size = train_size
        self._val_size = val_size
        self._test_size = test_size

        # Set up the parameters for the usage of the model
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

    def _restore_weights(self, experiment_path, log_type: str):
        log = self._test_log if log_type == 'testing' else self._train_log

        if self._epochs_params.initial_epoch < 10:
            init_epoch = '0' + str(self._epochs_params.initial_epoch)
        else:
            init_epoch = str(self._epochs_params.initial_epoch)
        restore_filename_reg = 'weights.{}-*.hdf5'.format(init_epoch)
        restore_path_reg = os.path.join(experiment_path, restore_filename_reg)
        list_files = glob.glob(restore_path_reg)
        assert len(list_files) > 0, 'ERR: No weights file match provided name {}'.format(
            restore_path_reg)

        # Take real filename
        restore_filename = list_files[0].split('/')[-1]
        restore_path = os.path.join(experiment_path, restore_filename)

        assert os.path.isfile(restore_path), \
            'ERR: Weight file in path {} seems not to be a file'.format(restore_path)
        log.info("Restoring weights in file {}...".format(restore_filename))
        self._model.load_weights(restore_path)

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
        Plot the keras model in png format
        """
        plot_model(self._model, to_file=os.path.join(self._current_experiment_path, 'model.png'))

    def display_summary(self):
        """
        Displays the architecture of the model
        """
        self._model.summary()

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
