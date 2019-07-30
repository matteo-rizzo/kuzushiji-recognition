import glob
import logging
import os
from typing import List

import tensorflow as tf
from tensorflow.python.keras import models, optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.python.keras.utils import plot_model


class Model:
    def __init__(self,
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 run_id: str,
                 ratios: {},
                 model_params: {}):
        """
        Builds a keras model and sets the train and test sets
        :param training_set: tf.Dataset object with train images
        :param training_set: tf.Dataset object with validation images
        :param test_set: tf.Dataset object with images for prediction
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
        self._model: models.Sequential = None

    def get_model(self) -> models.Sequential:
        return self._model

    def _build(self):
        pass

    def __restore_weights(self, experiment_path):
        log = logging.getLogger('training')

        initial_epoch = self._epochs_params.initial

        if initial_epoch < 10:
            init_epoch = '0' + str(initial_epoch)
        else:
            init_epoch = str(initial_epoch)

        restore_filename_reg = 'weights.{}-*.hdf5'.format(init_epoch)
        restore_path_reg = os.path.join(experiment_path, restore_filename_reg)
        list_files = glob.glob(restore_path_reg)
        assert len(list_files) > 0, 'ERR: No weights file match for provided name {}'.format(restore_path_reg)

        restore_filename = list_files[0].split('/')[-1]
        restore_path = os.path.join(experiment_path, restore_filename)

        assert os.path.isfile(restore_path), 'ERR: Weight file in path {} seems not to be a file'.format(restore_path)

        log.info('Restoring weights in file {}...'.format(restore_filename))
        self._model.load_weights(restore_path)

    def __compile_model(self):
        # Set the learning rate
        lr = self._network_params['learning_rate'] if self._network_params['learning_rate'] > 0.0 else 0.001
        lr = tf.train.exponential_decay(
            learning_rate=lr,
            global_step=self._epochs_params['initial'],
            decay_rate=self._network_params['lr_decay'],
            decay_steps=self._network_params['n_images'] * self._ratios['training'] //
                        self._network_params['batch_size']
        )

        # Set the optimizer
        optimizer = optimizers.Adam(lr=lr)
        self._model.compile(optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['sparse_categorical_accuracy'])
        return lr

    def __setup_callbacks(self):
        # Create a folder for the model log of the current experiment
        weights_log_path = os.path.join(self._current_experiment_path, 'weights')

        # Setup callback to save the best weights after each epoch
        checkpointer = ModelCheckpoint(filepath=os.path.join(weights_log_path,
                                                             'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min')

        tensorboard_log_dir = os.path.join(self._current_experiment_path, 'tensorboard')

        # Note that update_freq is set to batch_size * 10 because the epoch takes too long and batch size too short
        tensorboard = TensorBoard(log_dir=tensorboard_log_dir,
                                  write_graph=True,
                                  histogram_freq=0,
                                  write_grads=True,
                                  write_images=False,
                                  batch_size=self._network_params['batch_size'],
                                  update_freq=self._network_params['batch_size'] * 10)

        # setup early stopping to stop training if val_loss is not increasing after 3 epochs
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            mode='min',
        )
        return [early_stopping, tensorboard, checkpointer]

    def train(self):
        """
         Compile and train the model for the specified number of epochs.
         """

        log = logging.getLogger('training')

        log.info('Training the model...\n')

        # Set the number of epochs
        epochs = self._epochs_params['number']

        # Compile the model
        log.info('Compiling the model...')
        self.__compile_model()
        logging.info('Model compiled successfully!')

        # Create a folder for the model log of the current experiment
        weights_log_path = os.path.join(self._current_experiment_path, 'weights')
        os.makedirs(weights_log_path, exist_ok=True)

        # Restore weights if the proper flag has been set
        if self._epochs_params['restore_weights']:
            self.__restore_weights(weights_log_path)
        else:
            if len(os.listdir(weights_log_path)) > 0:
                raise FileExistsError(
                    '{} has trained weights. Please change run_id or delete existing folder.'.format(weights_log_path))

        # Display the architecture of the model
        log.info('Architecture of the model:')
        self.display_summary()

        log.info('Setting up the checkpointer...')
        callbacks = self.__setup_callbacks()

        # Train the model
        log.info('Starting the fitting procedure:')
        log.info('* Total number of epochs:   ' + str(self._epochs_params['number']))
        log.info('* Initial epoch:            ' + str(self._epochs_params['initial']) + '\n')
        self._model.fit(self._training_set,
                        epochs=epochs,
                        steps_per_epoch=int(self._ratios['training'] // self._network_params['batch_size']) + 1,
                        validation_data=self._validation_set,
                        validation_steps=int(self._ratios['validation'] // self._network_params['batch_size']) + 1,
                        callbacks=callbacks,
                        initial_epoch=int(self._epochs_params['initial']))

        # Set up a flag which states that the network is now trained and can be evaluated
        log.info('Training procedure performed successfully!\n')
        self._trained = True

    def plot_model(self):
        """
        Plot the keras model in png format
        """
        plot_model(self._model, to_file='model.png')

    def display_summary(self):
        """
        Displays the architecture of the model
        """
        self._model.summary()

    def evaluate(self) -> (float, float):
        """
        Evaluate the model returning loss and accuracy.
        :return: two lists of scalars, one for loss and one metrics
        """

        log = logging.getLogger('testing')
        log.info('Testing the model...')

        if not self._trained:
            # Create a folder for the model log of the current experiment
            weights_log_path = os.path.join(self._current_experiment_path, 'weights')
            self.__restore_weights(weights_log_path)

        # Compile the model
        log.info('Compiling the model...')
        _ = self.__compile_model()

        test_ratio = 1 - self._ratios.training - self._ratios.validation
        test_loss, test_acc = self._model.evaluate(self._test_set,
                                                   steps=test_ratio // self._network_params['batch_size'])
        return test_loss, test_acc

    def predict(self) -> List[float]:
        test_ratio = 1 - self._ratios.training - self._ratios.validation
        result = self._model.predict(self._test_set,
                                     steps=test_ratio // self._network_params['batch_size'])
        return result
