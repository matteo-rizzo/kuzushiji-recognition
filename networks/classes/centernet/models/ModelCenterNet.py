import glob
import os
from typing import Dict
from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from networks.classes.centernet.models.ModelGenerator import ModelGenerator
# from networks.classes.centernet.models.ModelGeneratorNew import ModelGenerator


class ModelCenterNet:

    def __init__(self, logs: Dict):
        self.__logs = logs

    def build_model(self, input_shape, mode: str, n_category: int = 1) -> tf.keras.Model:
        """
        Builds the network.

        :param input_shape: the shape of the input images
        :param mode: the type of model that must be generated
        :param n_category: the number of categories (possible classes). Defaults to 1 in order to detect the
        presence or absence of an object only (and not its label).
        :return: a Keras model
        """

        self.__logs['execution'].info('Building {} model...'.format(mode))

        return ModelGenerator().generate_model(input_shape, mode, n_category)

    @staticmethod
    def setup_callbacks(weights_log_path: str, batch_size: int) -> List[tf.keras.callbacks.Callback]:
        """
        Sets up the callbacks for the training of the model.
        """

        # Setup callback to save the best weights after each epoch
        checkpointer = ModelCheckpoint(filepath=os.path.join(weights_log_path,
                                                             'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='min')

        tensorboard_log_dir = os.path.join(weights_log_path, 'tensorboard')

        # Note that update_freq is set to batch_size * 10,
        # because the epoch takes too long and batch size too short
        tensorboard = TensorBoard(log_dir=tensorboard_log_dir,
                                  write_graph=True,
                                  histogram_freq=0,
                                  write_grads=True,
                                  write_images=False,
                                  batch_size=batch_size,
                                  update_freq=batch_size * 10)

        # Setup early stopping to stop training if val_loss is not increasing after 3 epochs
        # early_stopping = EarlyStopping(
        #    monitor='val_loss',
        #    patience=2,
        #    mode='min',
        # )

        return [tensorboard, checkpointer]

    def restore_weights(self,
                        model: tf.keras.Model,
                        init_epoch: int,
                        weights_folder_path: str) -> None:
        """
        Restores the weights from an existing weights file

        :param model:
        :param init_epoch:
        :param weights_folder_path:
        """

        init_epoch_str = '0' + str(init_epoch) if init_epoch < 10 else str(init_epoch)

        restore_path_reg = os.path.join(weights_folder_path, 'weights.{}-*.hdf5'.format(init_epoch_str))
        list_files = glob.glob(restore_path_reg)
        assert len(list_files) > 0, \
            'ERR: No weights file match provided name {}'.format(restore_path_reg)

        # Take real filename
        restore_filename = list_files[0].split('/')[-1]
        restore_path = os.path.join(weights_folder_path, restore_filename)
        assert os.path.isfile(restore_path), \
            'ERR: Weight file in path {} seems not to be a file'.format(restore_path)

        self.__logs['execution'].info("Restoring weights in file {}...".format(restore_filename))
        model.load_weights(restore_path)

    def train(self,
              model: tf.keras.Model,
              init_epoch: int,
              epochs: int,
              training_set: tf.data.Dataset,
              validation_set: tf.data.Dataset,
              training_steps: int,
              validation_steps: int,
              callbacks: List[tf.keras.callbacks.Callback]):
        """
        Compiles and trains the model for the specified number of epochs.
        """

        self.__logs['training'].info('Training the model...\n')

        # Display the architecture of the model
        self.__logs['training'].info('Architecture of the model:')
        model.summary()

        # Train the model
        self.__logs['training'].info('Starting the fitting procedure:')
        self.__logs['training'].info('* Total number of epochs:   ' + str(epochs))
        self.__logs['training'].info('* Initial epoch:            ' + str(init_epoch) + '\n')

        model.fit(training_set,
                  epochs=epochs,
                  steps_per_epoch=training_steps,
                  validation_data=validation_set,
                  validation_steps=validation_steps,
                  callbacks=callbacks,
                  initial_epoch=init_epoch)

        self.__logs['training'].info('Training procedure performed successfully!\n')

    def evaluate(self,
                 model: tf.keras.Model,
                 evaluation_set: tf.data.Dataset,
                 evaluation_steps: int) -> Union[float, List[float]]:
        """
        Evaluate the model on provided set.
        :return: the loss value if model has no other metrics, otw returns array with loss and metrics
        values.
        """

        self.__logs['training'].info('Evaluating the model...')

        return model.evaluate(evaluation_set,
                              steps=evaluation_steps)

        # predictions = model.predict(evaluation_set, steps=evaluation_steps)
        #
        # # True values
        # target_labels = []
        # batch_count = 0
        # for batch_samples in evaluation_set:
        #     batch_count += 1
        #     target_labels.extend(batch_samples[1].numpy())
        #     if batch_count == evaluation_steps:
        #         # Seen all examples, so exit the dataset iteration
        #         break
        #
        # plt.scatter(predictions, target_labels[:len(predictions)])
        # plt.title('---Letter_size/picture_size--- estimated vs target ', loc='center', fontsize=10)
        # plt.show()

    def predict(self,
                model: tf.keras.Model,
                dataset: tf.data.Dataset) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Performs a prediction on a given dataset
        """

        self.__logs['test'].info("Predicting...")

        return model.predict(dataset)
