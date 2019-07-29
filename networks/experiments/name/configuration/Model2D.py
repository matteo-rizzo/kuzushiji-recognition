import tensorflow as tf
from tensorflow.python.keras import layers, models

from networks.classes.Model import Model


class Model2D(Model):
    def __init__(self,
                 training_set: tf.data.Dataset,
                 validation_set: tf.data.Dataset,
                 test_set: tf.data.Dataset,
                 params: any) -> None:
        super().__init__(training_set, validation_set, test_set, params)
        self._build(params)

    def _build(self, params) -> None:
        """
        Builds the convolutional network.
        """
        img_size = params.image_size
        in_channels = params.in_channels
        out_channels = params.out_channels
        num_labels = params.num_labels
        bn_momentum = params.bn_momentum
        channels = [out_channels, out_channels * 2, out_channels * 4]

        self.model = models.Sequential()

        for i, c in enumerate(channels):
            self.model.add(layers.Conv2D(filters=c,
                                         kernel_size=(7, 7),
                                         input_shape=(img_size, img_size, in_channels),
                                         data_format='channels_last',
                                         padding='same',
                                         name='conv1_{i}'.format(i=i)),
                           )
            self.model.add(layers.BatchNormalization(momentum=bn_momentum, name='batch_norm_{i}'.format(i=i)))
            self.model.add(layers.Activation('selu', name='selu1_{i}'.format(i=i)))
            self.model.add(
                layers.MaxPool2D(pool_size=(3, 3),
                                 strides=(3, 3),
                                 name='max_pool_{i}'.format(i=i), padding='valid'
                                 ))

        self.model.add(layers.Flatten(name='flatt_1'))
        self.model.add(layers.Dense(units=128, name='last_linear', activation='linear'))
        self.model.add(layers.BatchNormalization(momentum=bn_momentum, name='batch_norm_last'))
        self.model.add(layers.Activation('selu', name='last_selu'))
        self.model.add(layers.Dense(units=num_labels, name='classifier', activation='sigmoid'))
