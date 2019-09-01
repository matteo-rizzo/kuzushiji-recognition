from tensorflow.python.keras.layers import Conv2DTranspose, UpSampling2D, BatchNormalization, \
    LeakyReLU, Concatenate, Conv2D, Add


def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    x_deep = Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(
        x_deep)
    x_deep = BatchNormalization()(x_deep)
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x = cbr(x, out_ch, 1, 1)
    return x


def cbr(x, out_layer, kernel, stride):
    x = Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def resblock(x_in, layer_n):
    x = cbr(x_in, layer_n, 3, 1)
    x = cbr(x, layer_n, 3, 1)
    x = Add()([x, x_in])
    return x
