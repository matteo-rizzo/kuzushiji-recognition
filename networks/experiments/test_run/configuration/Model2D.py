import tensorflow as tf
from tensorflow.python.keras import layers, models

from networks.classes.Model import Model


class Model2D(Model):
    def __init__(self,
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 run_id: str,
                 ratios: {},
                 model_params: {}) -> None:
        # Construct the super class
        super().__init__(training_set,
                         validation_set,
                         test_set,
                         run_id,
                         ratios,
                         model_params)

        # Build the specific model
        self._build()

    def _build(self):
        """
        Builds the convolutional network.
        """
        # Set up the main parameters of the model
        img_size = self._network_params['image_size']
        in_channels = self._network_params['in_channels']
        out_channels = self._network_params['out_channels']
        num_labels = self._network_params['n_labels']
        bn_momentum = self._network_params['bn_momentum']

        # Set up the progression of the output channels
        channels = [out_channels, out_channels * 2, out_channels * 4]

        # Initialize the model
        self._model = models.Sequential()

        for i, c in enumerate(channels):
            self._model.add(layers.Conv2D(filters=c,
                                          kernel_size=(7, 7),
                                          input_shape=(img_size, img_size, in_channels),
                                          data_format='channels_last',
                                          padding='same',
                                          name='conv1_{i}'.format(i=i)),
                            )
            self._model.add(layers.BatchNormalization(momentum=bn_momentum, name='batch_norm_{i}'.format(i=i)))
            self._model.add(layers.Activation('selu', name='selu1_{i}'.format(i=i)))
            self._model.add(
                layers.MaxPool2D(pool_size=(3, 3),
                                 strides=(3, 3),
                                 name='max_pool_{i}'.format(i=i), padding='valid'
                                 ))

        self._model.add(layers.Flatten(name='flatt_1'))
        self._model.add(layers.Dense(units=128, name='last_linear', activation='linear'))
        self._model.add(layers.BatchNormalization(momentum=bn_momentum, name='batch_norm_last'))
        self._model.add(layers.Activation('selu', name='last_selu'))
        self._model.add(layers.Dense(units=num_labels, name='classifier', activation='sigmoid'))
