import copy
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
                 params: any) -> None:
        """
        Builds a keras model and sets the train and test sets
        :param test_set: tf.Dataset object with images for prediction
        :param training_set: tf.Dataset object with train images
        """
        self.trained = False
        self.test_set = test_set
        self.training_set = training_set
        self.validation_set = validation_set
        self.params = params
        # Set the path to the current experiment
        self.current_experiment_path = os.path.join(os.getcwd(), 'networks', 'experiments',
                                                    params.run_id)
        self.model: models.Sequential = None

    def get_model(self) -> models.Sequential:
        return self.model

    def drop_layer(self, layer: models.Sequential.layers):
        weights = layer.get_weights()
        zeros = copy.deepcopy(weights)
        for zero in zeros:
            zero.fill(0)
        self.model.get_layer(layer.name).set_weights(zeros)

        return weights

    def restore_layer(self, layer: models.Sequential.layers, weights):
        self.model.get_layer(layer.name).set_weights(weights)

    def _build(self, params) -> None:
        pass

    def __restore_weights(self, experiment_path, log_type):
        log = logging.getLogger(log_type)
        # List file by
        if self.params.initial_epoch < 10:
            init_epoch = '0' + str(self.params.initial_epoch)
        else:
            init_epoch = str(self.params.initial_epoch)
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
        self.model.load_weights(restore_path)

    def __compile_model(self):
        # Set the learning rate
        lr = self.params.learning_rate if self.params.learning_rate > 0.0 else 0.001

        lr = tf.train.exponential_decay(
            learning_rate=lr,
            global_step=self.params.initial_epoch,
            decay_rate=self.params.lr_decay,
            decay_steps=self.params.n_images * self.params.training_ratio // self.params.batch_size

        )
        # Set the optimizer
        optimizer = optimizers.Adam(lr=lr)
        self.model.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['sparse_categorical_accuracy'])
        return lr

    def __setup_callbacks(self):
        # Create a folder for the model log of the current experiment
        weights_log_path = os.path.join(self.current_experiment_path, 'weights')
        # Setup callback to save the best weights after each epoch
        checkpointer = ModelCheckpoint(filepath=os.path.join(weights_log_path,
                                                             'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min')

        tensorboard_log_dir = os.path.join(self.current_experiment_path, 'tensorboard')
        # Note that update_freq is set to batch_size * 10 because the epoch takes too long and batch size too short
        tensorboard = TensorBoard(log_dir=tensorboard_log_dir,
                                  write_graph=True,
                                  histogram_freq=0,
                                  write_grads=True,
                                  write_images=False,
                                  batch_size=self.params.batch_size,
                                  update_freq=self.params.batch_size * 10)

        # setup early stopping to stop training if val_loss is not increasing after 3 epochs
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            mode='min',
        )
        return [early_stopping, tensorboard, checkpointer]

    def train(self) -> None:
        """
         Compile and train the model for the specified number of epochs.
         :param run_id: the identification code of the current experiment
         :param params: object with number of epochs, restore option, batch size and training set size
         """

        log = logging.getLogger('training')

        log.info("Training the model...\n")

        # Set the number of epochs
        epochs = self.params.epochs

        # Compile the model
        log.info("Compiling the model...")
        lr = self.__compile_model()

        log.info("Main parameters:")
        log.info("* Number of epochs:   " + str(epochs))
        log.info("* Base learning rate:      " + str(lr) + '\n')

        # Create a folder for the model log of the current experiment
        weights_log_path = os.path.join(self.current_experiment_path, 'weights')
        os.makedirs(weights_log_path, exist_ok=True)

        # Restore weights if the proper flag has been set
        if self.params.restore_weights:
            self.__restore_weights(weights_log_path, 'training')
        else:
            if len(os.listdir(weights_log_path)) > 0:
                raise FileExistsError(
                    "{} has trained weights. Please change run_id or delete existing folder.".format(
                        weights_log_path))

        logging.info("Model compiled successfully!")

        # Display the architecture of the model
        log.info("Architecture of the model:")
        self.display_summary()

        log.info("Setting up the checkpointer...")
        callbacks = self.__setup_callbacks()

        # Train the model
        log.info("Starting the fitting procedure...")
        history = self.model.fit(self.training_set,
                                 epochs=epochs,
                                 steps_per_epoch=self.params.training_size // self.params.batch_size,
                                 validation_data=self.validation_set,
                                 validation_steps=self.params.validation_size // self.params.batch_size,
                                 callbacks=callbacks,
                                 initial_epoch=self.params.initial_epoch)

        # Set up a flag which states that the network is now trained and can be evaluated
        log.info("Training done successfully!\n")
        self.trained = True

    def plot_model(self):
        """
        Plot the keras model in png format
        """
        plot_model(self.model, to_file='model.png')

    def display_summary(self):
        """
        Displays the architecture of the model
        """
        self.model.summary()

    def evaluate(self) -> (float, float):
        """
        Evaluate the model returning loss and accuracy.
        :return: two lists of scalars, one for loss and one metrics
        """

        log = logging.getLogger('testing')
        log.info("Testing the model...")
        if not self.trained:
            # Create a folder for the model log of the current experiment
            weights_log_path = os.path.join(self.current_experiment_path, 'weights')
            self.__restore_weights(weights_log_path, 'testing')

        # Compile the model
        log.info("Compiling the model...")
        _ = self.__compile_model()

        test_loss, test_acc = self.model.evaluate(self.test_set,
                                                  steps=self.params.test_size // self.params.batch_size)
        return test_loss, test_acc

    def predict(self, params: any) -> List[float]:
        """
        THIS METHOD GIVE A PREDICTION OVER SOME ITEMS. THE RESULT IS AN ARRAY, ONE ELEMENT PER PREDICTION. THIS IS
        NOT USEFUL TO TEST THE MODEL BUT ONLY FOR PRODUCTION.
        :param params: object with test set size and batch size
        :return: the numpy array with predictions
        """
        result = self.model.predict(self.test_set, steps=params.test_size // params.batch_size)
        return result
