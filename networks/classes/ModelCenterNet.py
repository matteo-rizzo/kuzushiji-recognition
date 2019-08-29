import glob
import logging
import os
from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import UpSampling2D, Concatenate, Conv2D, Input, AveragePooling2D, \
    GlobalAveragePooling2D, Dense, Dropout, Activation

from networks.functions.blocks import cbr, aggregation_block, resblock


class ModelUtilities:

    @staticmethod
    def __resize_input_layers(input_layer):
        input_layer_1 = AveragePooling2D(2)(input_layer)
        input_layer_2 = AveragePooling2D(2)(input_layer_1)

        return input_layer_1, input_layer_2

    @staticmethod
    def __generate_encoder(input_layer_0, input_layer_1, input_layer_2):

        # 512->256
        x_0 = cbr(input_layer_0, 16, 3, 2)
        concat_1 = Concatenate()([x_0, input_layer_1])

        # 256->128
        x_1 = cbr(concat_1, 32, 3, 2)
        concat_2 = Concatenate()([x_1, input_layer_2])

        # 128->64
        x_2 = cbr(concat_2, 64, 3, 2)

        x = cbr(x_2, 64, 3, 1)
        x = resblock(x, 64)
        x = resblock(x, 64)

        # 64->32
        x_3 = cbr(x, 128, 3, 2)
        x = cbr(x_3, 128, 3, 1)
        x = resblock(x, 128)
        x = resblock(x, 128)
        x = resblock(x, 128)

        # 32->16
        x_4 = cbr(x, 256, 3, 2)
        x = cbr(x_4, 256, 3, 1)
        x = resblock(x, 256)
        x = resblock(x, 256)
        x = resblock(x, 256)
        x = resblock(x, 256)
        x = resblock(x, 256)

        # 16->8
        x_5 = cbr(x, 512, 3, 2)
        x = cbr(x_5, 512, 3, 1)

        x = resblock(x, 512)
        x = resblock(x, 512)
        x = resblock(x, 512)

        return x_1, x_2, x_3, x_4, x

    def __generate_preprocessing_model(self, input_layer, n_category=None):

        input_layer_1, input_layer_2 = self.__resize_input_layers(input_layer)

        _, _, _, _, x = self.__generate_encoder(input_layer, input_layer_1, input_layer_2)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)

        out = Dense(1, activation="linear")(x)

        return Model(input_layer, out)

    def __generate_centernet_model(self, input_layer, n_category=None):

        output_layer_n = 1 + 4

        input_layer_1, input_layer_2 = self.__resize_input_layers(input_layer)

        x_1, x_2, x_3, x_4, x = self.__generate_encoder(input_layer, input_layer_1, input_layer_2)

        x_1 = cbr(x_1, output_layer_n, 1, 1)
        x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

        x_2 = cbr(x_2, output_layer_n, 1, 1)
        x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)

        x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

        x_3 = cbr(x_3, output_layer_n, 1, 1)
        x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n)
        x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
        x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

        x_4 = cbr(x_4, output_layer_n, 1, 1)

        x = cbr(x, output_layer_n, 1, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 8->16 tconv
        x = Concatenate()([x, x_4])
        x = cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 16->32
        x = Concatenate()([x, x_3])

        x = cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 32->64   128
        x = Concatenate()([x, x_2])
        x = cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 64->128
        x = Concatenate()([x, x_1])

        x = Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)
        # x = MaxPooling2D(pool_size=(3, 3), strides=None, padding="same")(x)

        out = Activation("sigmoid")(x)

        return Model(input_layer, out)

    @staticmethod
    def __generate_classification_model(input_layer, n_category):

        x = cbr(input_layer, 64, 3, 1)
        x = resblock(x, 64)
        x = resblock(x, 64)
        x = cbr(input_layer, 128, 3, 2)  # 16
        x = resblock(x, 128)
        x = resblock(x, 128)
        x = cbr(input_layer, 256, 3, 2)  # 8
        x = resblock(x, 256)
        x = resblock(x, 256)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)

        out = Dense(n_category, activation="softmax")(x)

        return Model(input_layer, out)

    def generate_model(self, input_shape, mode: int, n_category: int = 1) -> tf.keras.Model:
        """
        Builds the network.

        :param input_shape: the shape of the input images
        :param mode: the type of model that must be generated
        :param n_category: the number of categories (possible classes). Defaults to 1 in order to detect the
        presence or absence of an object only (and not its label).
        :return: a Keras model
        """

        modes = {
            '1': self.__generate_preprocessing_model,
            '2': self.__generate_centernet_model,
            '3': self.__generate_classification_model
        }

        input_layer = Input(input_shape)

        return modes[str(mode)](input_layer, n_category)

    @staticmethod
    def setup_callbacks(weights_log_path: str, batch_size: int) \
            -> List[tf.keras.callbacks.Callback]:
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

        # Note that update_freq is set to batch_size * 10 because the epoch takes too long and batch size too short
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

    @staticmethod
    def restore_weights(model: tf.keras.Model, logger: logging.Logger, init_epoch: int,
                        weights_folder_path: str) -> None:
        if init_epoch < 10:
            init_epoch_str = '0' + str(init_epoch)
        else:
            init_epoch_str = str(init_epoch)

        restore_filename_reg = 'weights.{}-*.hdf5'.format(init_epoch_str)
        restore_path_reg = os.path.join(weights_folder_path, restore_filename_reg)
        list_files = glob.glob(restore_path_reg)
        assert len(list_files) > 0, 'ERR: No weights file match provided name {}'.format(
            restore_path_reg)

        # Take real filename
        restore_filename = list_files[0].split('/')[-1]
        restore_path = os.path.join(weights_folder_path, restore_filename)

        assert os.path.isfile(restore_path), \
            'ERR: Weight file in path {} seems not to be a file'.format(restore_path)
        logger.info("Restoring weights in file {}...".format(restore_filename))

        model.load_weights(restore_path)

    @staticmethod
    def train(model: tf.keras.Model,
              logger: logging.Logger,
              init_epoch: int,
              epochs: int,
              training_set: tf.data.Dataset,
              validation_set: tf.data.Dataset,
              training_steps: int,
              validation_steps: int,
              callbacks: List[tf.keras.callbacks.Callback]):

        """
        Compile and train the model for the specified number of epochs.
        """

        logger.info('Training the model...\n')

        # Display the architecture of the model
        logger.info('Architecture of the model:')
        model.summary()

        # Train the model
        logger.info('Starting the fitting procedure:')
        logger.info('* Total number of epochs:   ' + str(epochs))
        logger.info('* Initial epoch:            ' + str(init_epoch) + '\n')

        model.fit(training_set,
                  epochs=epochs,
                  steps_per_epoch=training_steps,
                  validation_data=validation_set,
                  validation_steps=validation_steps,
                  callbacks=callbacks,
                  initial_epoch=init_epoch)

        # Set up a flag which states that the network is now trained and can be evaluated
        logger.info('Training procedure performed successfully!\n')

    @staticmethod
    def evaluate(model: tf.keras.Model, logger: logging.Logger, evaluation_set: tf.data.Dataset,
                 evaluation_steps: int) -> Union[float, List[float]]:
        """
        Evaluate the model on provided set.
        :return: the loss value if model has no other metrics, otw returns array with loss and metrics
        values.
        """

        logger.info('Evaluating the model...')

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

    @staticmethod
    def predict(model: tf.keras.Model, logger: logging.Logger, dataset: tf.data.Dataset, steps: int) \
            -> Union[np.ndarray, List[np.ndarray]]:
        logger.info("Predicting...")

        result = model.predict(dataset, steps=steps)

        return result
