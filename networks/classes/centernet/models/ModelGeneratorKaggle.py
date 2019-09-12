import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2DTranspose, BatchNormalization, \
    LeakyReLU, Concatenate, Conv2D, Add
from tensorflow.python.keras.layers import UpSampling2D, Input, AveragePooling2D, \
    GlobalAveragePooling2D, Dense, Dropout, Activation


class ModelGeneratorKaggle:

    def __aggregation_block(self, x_shallow, x_deep, deep_ch, out_ch):
        x_deep = Conv2DTranspose(deep_ch,
                                 kernel_size=2,
                                 strides=2,
                                 padding='same',
                                 use_bias=False)(x_deep)
        x_deep = BatchNormalization()(x_deep)
        x_deep = LeakyReLU(alpha=0.1)(x_deep)

        x = Concatenate()([x_shallow, x_deep])
        x = self.__cbr(x, out_ch, 1, 1)

        return x

    def __alt_res_block(self, x_in, filter_n):
        x = self.__cbr(x_in, filter_n, 3, 1)
        x = self.__cbr(x, filter_n, 3, 1)
        x = Add()([x, x_in])

        return x

    @staticmethod
    def __cbr(x, filter_n, kernel, strides):
        x = Conv2D(filter_n, kernel_size=kernel, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

        return x

    @staticmethod
    def __resize_input_layers(input_layer):
        input_layer_1 = AveragePooling2D(2)(input_layer)
        input_layer_2 = AveragePooling2D(2)(input_layer_1)

        return input_layer_1, input_layer_2

    def __generate_encoder(self, input_layer_0, input_layer_1, input_layer_2):
        # 512->256
        x_0 = self.__cbr(input_layer_0, 16, 3, 2)
        concat_1 = Concatenate()([x_0, input_layer_1])

        # 256->128
        x_1 = self.__cbr(concat_1, 32, 3, 2)
        concat_2 = Concatenate()([x_1, input_layer_2])

        # 128->64
        x_2 = self.__cbr(concat_2, 64, 3, 2)

        x = self.__cbr(x_2, 64, 3, 1)
        x = self.__alt_res_block(x, 64)
        x = self.__alt_res_block(x, 64)

        # 64->32
        x_3 = self.__cbr(x, 128, 3, 2)
        x = self.__cbr(x_3, 128, 3, 1)
        x = self.__alt_res_block(x, 128)
        x = self.__alt_res_block(x, 128)
        x = self.__alt_res_block(x, 128)

        # 32->16
        x_4 = self.__cbr(x, 256, 3, 2)
        x = self.__cbr(x_4, 256, 3, 1)
        x = self.__alt_res_block(x, 256)
        x = self.__alt_res_block(x, 256)
        x = self.__alt_res_block(x, 256)
        x = self.__alt_res_block(x, 256)
        x = self.__alt_res_block(x, 256)

        # 16->8
        x_5 = self.__cbr(x, 512, 3, 2)
        x = self.__cbr(x_5, 512, 3, 1)

        x = self.__alt_res_block(x, 512)
        x = self.__alt_res_block(x, 512)
        x = self.__alt_res_block(x, 512)

        return x_1, x_2, x_3, x_4, x

    def __generate_detection_model(self, input_layer, n_category=None):
        output_layer_n = n_category + 4

        input_layer_1, input_layer_2 = self.__resize_input_layers(input_layer)

        x_1, x_2, x_3, x_4, x = self.__generate_encoder(input_layer, input_layer_1, input_layer_2)

        x_1 = self.__cbr(x_1, output_layer_n, 1, 1)
        x_1 = self.__aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

        x_2 = self.__cbr(x_2, output_layer_n, 1, 1)
        x_2 = self.__aggregation_block(x_2, x_3, output_layer_n, output_layer_n)

        x_1 = self.__aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

        x_3 = self.__cbr(x_3, output_layer_n, 1, 1)
        x_3 = self.__aggregation_block(x_3, x_4, output_layer_n, output_layer_n)
        x_2 = self.__aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
        x_1 = self.__aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
        x_4 = self.__cbr(x_4, output_layer_n, 1, 1)

        x = self.__cbr(x, output_layer_n, 1, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 8->16
        x = Concatenate()([x, x_4])
        x = self.__cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 16->32
        x = Concatenate()([x, x_3])

        x = self.__cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 32->64
        x = Concatenate()([x, x_2])
        x = self.__cbr(x, output_layer_n, 3, 1)
        x = UpSampling2D(size=(2, 2))(x)  # 64->128
        x = Concatenate()([x, x_1])

        x = Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)

        out = Activation("sigmoid")(x)

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

        out = Dense(n_category, activation="softmax")(x)

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
            'detection': self.__generate_detection_model,
            'classification': self.__generate_classification_model
        }

        input_layer = Input(input_shape)

        return modes[str(mode)](input_layer, n_category)
