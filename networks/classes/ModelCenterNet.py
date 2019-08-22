import os
import glob
import tensorflow as tf
from typing import Tuple

from networks.classes.Logger import Logger
from networks.classes.Model import Model as MyModel
from tensorflow.python.keras import layers, models, optimizers

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from networks.functions.blocks import cbr, aggregation_block, resblock
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2DTranspose, UpSampling2D, BatchNormalization, \
    LeakyReLU, Concatenate, Conv2D, Add, Input, AveragePooling2D, GlobalAveragePooling2D, Dense, \
    Dropout, Activation

import matplotlib.pyplot as plt

output_layer_n = 1 + 4


class ModelCenterNet(MyModel):
    def __init__(self,
                 run_id: str,
                 model_params: {},
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 train_size: int,
                 val_size: int,
                 test_size: int,
                 input_shape: Tuple[int, int, int],
                 log_handler: Logger,
                 mode="sizecheck"):
        # Construct the super class
        super().__init__(run_id,
                         model_params,
                         training_set,
                         validation_set,
                         test_set,
                         train_size,
                         val_size,
                         test_size,
                         log_handler)

        self._size_detection_mode = mode

        # Build the custom model
        self._model: tf.python.keras.Model
        self._build(input_shape=input_shape)

    def _build(self, input_shape, aggregation=True):
        """
        Builds the network.
        """

        input_layer = Input(input_shape)

        # Resized input
        input_layer_1 = AveragePooling2D(2)(input_layer)
        input_layer_2 = AveragePooling2D(2)(input_layer_1)

        #### ENCODER ####

        x_0 = cbr(input_layer, 16, 3, 2)  # 512->256
        concat_1 = Concatenate()([x_0, input_layer_1])

        x_1 = cbr(concat_1, 32, 3, 2)  # 256->128
        concat_2 = Concatenate()([x_1, input_layer_2])

        x_2 = cbr(concat_2, 64, 3, 2)  # 128->64

        x = cbr(x_2, 64, 3, 1)
        x = resblock(x, 64)
        x = resblock(x, 64)

        x_3 = cbr(x, 128, 3, 2)  # 64->32
        x = cbr(x_3, 128, 3, 1)
        x = resblock(x, 128)
        x = resblock(x, 128)
        x = resblock(x, 128)

        x_4 = cbr(x, 256, 3, 2)  # 32->16
        x = cbr(x_4, 256, 3, 1)
        x = resblock(x, 256)
        x = resblock(x, 256)
        x = resblock(x, 256)
        x = resblock(x, 256)
        x = resblock(x, 256)

        x_5 = cbr(x, 512, 3, 2)  # 16->8
        x = cbr(x_5, 512, 3, 1)

        x = resblock(x, 512)
        x = resblock(x, 512)
        x = resblock(x, 512)

        if self._size_detection_mode:
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            out = Dense(1, activation="linear")(x)

        else:  # CenterNet mode
            #### DECODER ####
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
            out = Activation("sigmoid")(x)

        self._model = Model(input_layer, out)

    def _compile_model(self):
        lr = self._network_params["learning_rate"] if \
            self._network_params["learning_rate"] > 0.0 else 0.001

        # Set the optimizer
        optimizer = optimizers.Adam(lr=lr)
        self._model.compile(optimizer=optimizer,
                            loss='mean_squared_error')
        return lr

    def _setup_callbacks(self):
        """
        Sets up the callbacks for the training of the model.
        """
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
        # early_stopping = EarlyStopping(
        #    monitor='val_loss',
        #    patience=2,
        #    mode='min',
        # )

        return [tensorboard, checkpointer]

    def train(self):
        """
        Compile and train the model for the specified number of epochs.
        """

        self._train_log.info('Training the model...\n')

        # Set the number of epochs
        n_epochs = self._epochs_params['number']

        # Compile the model
        self._train_log.info('Compiling the model...')
        self._compile_model()
        self._train_log.info('Model compiled successfully!')

        # Create a folder for the model log of the current experiment
        weights_log_path = os.path.join(self._current_experiment_path, 'weights')
        os.makedirs(weights_log_path, exist_ok=True)

        # Restore weights if the proper flag has been set
        if self._epochs_params['restore_weights']:
            self._restore_weights(weights_log_path, 'training')
        else:
            if len(os.listdir(weights_log_path)) > 0:
                raise FileExistsError(
                    '{} has trained weights. Please change run_id or delete existing folder.'.format(
                        weights_log_path))

        # Display the architecture of the model
        self._train_log.info('Architecture of the model:')
        self.display_summary()

        self._train_log.info('Setting up the checkpointer...')
        callbacks = self._setup_callbacks()

        # Train the model
        self._train_log.info('Starting the fitting procedure:')
        self._train_log.info('* Total number of epochs:   ' + str(self._epochs_params['number']))
        self._train_log.info('* Initial epoch:            ' + str(self._epochs_params['initial']) + '\n')

        self._model.fit(self._training_set,
                        epochs=int(n_epochs),
                        steps_per_epoch=int(self._train_size // self._network_params['batch_size']) + 1,
                        validation_data=self._validation_set,
                        validation_steps=int(self._val_size // self._network_params['batch_size']) + 1,
                        callbacks=callbacks,
                        initial_epoch=int(self._epochs_params['initial']))

        # Set up a flag which states that the network is now trained and can be evaluated
        self._train_log.info('Training procedure performed successfully!\n')
        self._trained = True

    def evaluate(self) -> any:
        """
        Evaluate the model on the validation set.
        :return:
        """
        self._test_log.info("Evaluating the model...")
        if not self._trained:
            weights_log_path = os.path.join(self._current_experiment_path, 'weights')
            self._restore_weights(weights_log_path, 'testing')

        # Compile the model
        self._test_log.info("Compiling the model...")
        self._compile_model()

        predictions = self._model.predict(self._validation_set, steps=self._val_size //
                                                                      self._network_params['batch_size'])

        # test_loss, test_acc = self._model.evaluate(self._validation_set,
        #                                           steps=self._test_size //
        #                                           self._network_params['batch_size'])

        # True values
        target = tf.data.get_output_classes(self._validation_set)

        # BISOGNA CAPIRE COSA RITORNA in TARGET

        plt.scatter(predictions, target[:len(predictions)])
        plt.title('---letter_size/picture_size--- estimated vs target ', loc='center', fontsize=10)
        plt.show()

        # return test_loss, test_acc

    def predict(self) -> [float]:
        assert self._trained, "Error: can't predict with untrained model!"

        result = self._model.predict(self._test_set,
                                     steps=self._test_size // self._network_params['batch_size'] + 1)
        return result
