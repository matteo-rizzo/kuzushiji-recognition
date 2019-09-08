import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2DTranspose, BatchNormalization, LeakyReLU, Concatenate, \
    Conv2D, Add
from tensorflow.python.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Activation, \
    ZeroPadding2D, \
    MaxPooling2D

from tensorflow.python.keras.applications.resnet50 import ResNet50


class ModelGeneratorTile:

    @staticmethod
    def __cbr(x, filter_n, kernel, strides):
        x = Conv2D(filter_n, kernel_size=kernel, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        return x

    # Residual blocks

    def __preactivated_res_block(self, x_in, filter_n):
        x = BatchNormalization()(x_in)
        x = LeakyReLU(alpha=0.1)(x)
        x = self.__cbr(x, filter_n, 3, 1)
        x = Conv2D(filter_n, kernel_size=3, strides=1, padding='same')(x)
        x = Add()([x, x_in])

        return x

    def __alt_res_block(self, x_in, filter_n):
        x = self.__cbr(x_in, filter_n, 3, 1)
        x = self.__cbr(x, filter_n, 3, 1)
        x = Add()([x, x_in])

        return x

    def __res_block(self, x_in, filter_n, strides: int = 1):
        z = x_in

        x = self.__cbr(x_in, filter_n, 3, strides)

        x = Conv2D(filter_n, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        # Projection shortcut
        if strides != 1 or strides != (1, 1):
            z = Conv2D(filter_n, kernel_size=1, strides=strides, padding='same')(x_in)
            z = BatchNormalization()(z)

        x = Add()([z, x])
        x = LeakyReLU(alpha=0.1)(x)

        return x

    def __generate_encoder(self, input_layer):
        # Block 1: (512, 512, 3) -> (128, 128, 64)
        x = ZeroPadding2D(3)(input_layer)
        x = Conv2D(64, kernel_size=7, strides=2, padding='valid')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        # Block 2: (128, 128, 64) -> (128, 128, 64)
        x = self.__res_block(x, filter_n=64, strides=1)
        x = self.__res_block(x, filter_n=64, strides=1)
        x_1 = self.__res_block(x, filter_n=64, strides=1)

        # Block 3: (128, 128, 64) -> (64, 64, 128)
        x = self.__res_block(x_1, filter_n=128, strides=2)  # projection shortcut
        x = self.__res_block(x, filter_n=128, strides=1)
        x = self.__res_block(x, filter_n=128, strides=1)
        x_2 = self.__res_block(x, filter_n=128, strides=1)

        # Block 4: (64, 64, 128) -> (32, 32, 256)
        x = self.__res_block(x_2, filter_n=256, strides=2)  # projection shortcut
        x = self.__res_block(x, filter_n=256, strides=1)
        x = self.__res_block(x, filter_n=256, strides=1)
        x = self.__res_block(x, filter_n=256, strides=1)
        x = self.__res_block(x, filter_n=256, strides=1)
        x_3 = self.__res_block(x, filter_n=256, strides=1)

        # Block 5: (64, 64, 128) -> (16, 16, 512)
        x = self.__res_block(x_3, filter_n=512, strides=2)  # projection shortcut
        x = self.__res_block(x, filter_n=512, strides=1)
        x = self.__res_block(x, filter_n=512, strides=1)

        return x_1, x_2, x_3, x

    @staticmethod
    def __generate_pretrained_encoder(input_layer: Model):
        resnet = ResNet50(include_top=False, weights='imagenet',
                          input_tensor=input_layer, input_shape=(512, 512, 3),
                          pooling=None)

        return resnet

    def __generate_detection_model_2(self, input_layer, n_category=None):
        out_filters = n_category + 4

        resnet: Model = self.__generate_pretrained_encoder(input_layer)

        x_0 = resnet.get_layer("activation").output
        x_1 = resnet.get_layer("activation_9").output
        x_2 = resnet.get_layer("activation_21").output
        x_3 = resnet.get_layer("activation_39").output
        x = resnet.get_layer("activation_48").output

        # Deconvolution block 1: (16, 16, 512) -> (32, 32, 512)
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
        x = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(x)
        x = Concatenate()([x_3, x])
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Deconvolution block 2: (32, 32, 1024) -> (64, 64, 256)
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
        x = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(x)
        x = Concatenate()([x_2, x])
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Deconvolution block 3: (64, 64, 256) -> (128, 128, 128)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(x)
        x = Concatenate()([x_1, x])
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Block 4:
        x = Conv2D(filters=out_filters, kernel_size=1, strides=1, padding='same')(x)
        out = Activation('sigmoid')(x)  # optional

        return Model(input_layer, out)

    def __generate_preprocessing_model(self, input_layer, n_category=None):
        # input_layer_1, input_layer_2 = self.__resize_input_layers(input_layer)

        _, _, _, x = self.__generate_encoder(input_layer)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)

        out = Dense(1, activation="linear")(x)

        return Model(input_layer, out)

    def __generate_detection_model(self, input_layer, n_category=None):
        out_filters = n_category + 4

        x_1, x_2, x_3, x = self.__generate_encoder(input_layer)

        # Deconvolution block 1: (16, 16, 512) -> (32, 32, 512)
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
        x = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(x)
        x = Concatenate()([x_3, x])
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.2)(x)

        # Deconvolution block 2: (32, 32, 512) -> (64, 64, 256)
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
        x = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(x)
        x = Concatenate()([x_2, x])
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.2)(x)

        # Deconvolution block 3: (64, 64, 256) -> (128, 128, 128)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(x)
        x = Concatenate()([x_1, x])
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Block 4:
        x = Conv2D(filters=out_filters, kernel_size=1, strides=1, padding='same')(x)
        out = Activation('sigmoid')(x)  # optional

        return Model(input_layer, out)

    def __generate_classification_model(self, input_layer, n_category):
        x = self.__cbr(input_layer, 64, 3, 1)
        x = self.__alt_res_block(x, 64)
        x = self.__alt_res_block(x, 64)

        x = self.__cbr(x, 128, 3, 2)  # 16
        x = self.__alt_res_block(x, 128)
        x = self.__alt_res_block(x, 128)

        x = self.__cbr(x, 256, 3, 2)  # 8
        x = self.__alt_res_block(x, 256)
        x = self.__alt_res_block(x, 256)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)

        out = Dense(n_category, activation='softmax')(x)

        return Model(input_layer, out)

    def generate_model(self, input_shape, mode: str, n_category: int = 1) -> tf.keras.Model:
        """
        Builds the network.

        :param input_shape: the shape of the input images
        :param mode: the type of model that must be generated
        :param n_category: the number of categories (possible classes). Defaults to 1 in order to detect the
        presence or absence of an object only (and not its label).
        :return: a Keras model
        """

        modes = {
            'preprocessing': self.__generate_preprocessing_model,
            'detection': self.__generate_detection_model,
            'classification': self.__generate_classification_model
        }

        input_layer = Input(input_shape)

        return modes[str(mode)](input_layer, n_category)
