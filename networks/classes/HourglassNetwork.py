import datetime
import os

import keras
import scipy.misc
from data_process import normalize
from eval_heatmap import cal_heatmap_acc
from keras.callbacks import CSVLogger
from keras.layers import *
from keras.losses import mean_squared_error
from keras.models import *
from keras.optimizers import RMSprop
from mpii_datagen import MPIIDataGen


class HourglassNetwork:

    def __init__(self, num_classes, num_stacks, num_channels, in_res, out_res):
        self.__num_classes = num_classes
        self.__num_stacks = num_stacks
        self.__num_channels = num_channels
        self.__in_res = in_res
        self.__out_res = out_res

        self.__model = None
        self.__build()

    def train(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    in_res=self.__in_res,
                                    out_res=self.__out_res,
                                    is_train=True)
        train_gen = train_dataset.generator(batch_size,
                                            self.__num_stacks,
                                            sigma=1,
                                            is_shuffle=True,
                                            rot_flag=True,
                                            scale_flag=True,
                                            flip_flag=True)

        csv_logger = CSVLogger(os.path.join(model_path,
                                            "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        model_file = os.path.join(model_path, 'weights_{epoch:02d}_{loss:.2f}.hdf5')

        checkpoint = self.EvalCallBack(model_path, self.__in_res, self.__out_res)

        x_callbacks = [csv_logger, checkpoint]

        self.__model.fit_generator(generator=train_gen,
                                   steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                   epochs=epochs,
                                   callbacks=x_callbacks)

    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):

        self.load_model(model_json, model_weights)
        self.__model.compile(optimizer=RMSprop(lr=5e-4),
                             loss=mean_squared_error,
                             metrics=["accuracy"])

        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    in_res=self.__in_res,
                                    out_res=self.__out_res,
                                    is_train=True)

        train_gen = train_dataset.generator(batch_size,
                                            self.__num_stacks,
                                            sigma=1,
                                            is_shuffle=True,
                                            rot_flag=True,
                                            scale_flag=True,
                                            flip_flag=True)

        model_dir = os.path.dirname(os.path.abspath(model_json))
        print(model_dir, model_json)
        csv_logger = CSVLogger(os.path.join(model_dir,
                                            "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        checkpoint = self.EvalCallBack(model_dir, self.__in_res, self.__out_res)

        x_callbacks = [csv_logger, checkpoint]

        self.__model.fit_generator(generator=train_gen,
                                   steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                   initial_epoch=init_epoch,
                                   epochs=epochs,
                                   callbacks=x_callbacks)

    def load_model(self, model_json, model_file):
        with open(model_json) as f:
            self.__model = model_from_json(f.read())

        self.__model.load_weights(model_file)

    def inference_rgb(self, rgb_data, org_shape, mean=None):

        scale = (org_shape[0] * 1.0 / self.__in_res[0], org_shape[1] * 1.0 / self.__in_res[1])
        img_data = scipy.misc.imresize(rgb_data, self.__in_res)

        if mean is None:
            mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

        img_data = normalize(img_data, mean)

        input = img_data[np.newaxis, :, :, :]

        out = self.__model.predict(input)
        return out[-1], scale

    def inference_file(self, img_file, mean=None):
        img_data = scipy.misc.imread(img_file)

        return self.inference_rgb(img_data, img_data.shape, mean)

    def display_summary(self):
        """
        Displays the architecture of the model
        """
        self.__model.summary()

    def __build(self, mobile=False):
        """
        Builds an Hourglass network

        :param mobile: specifies if the network is mobile type
        """

        if mobile:
            self.__model = self.__create_hourglass_network(self.__bottleneck_mobile)
        else:
            self.__model = self.__create_hourglass_network(self.__bottleneck_block)

    def __create_hourglass_network(self, bottleneck):
        """
        Creates the various layers of the network
        
        :param bottleneck: a function to be called in order to create the specific type of bottleneck 
        :return: 
        """

        input_layer = Input(shape=(self.__in_res[0], self.__in_res[1], 3))

        front_features = self.__create_front_module(input_layer, bottleneck)

        head_next_stage = front_features

        outputs = []
        for i in range(self.__num_stacks):
            head_next_stage, head_to_loss = self.__hourglass_module(bottom=head_next_stage,
                                                                    bottleneck=bottleneck,
                                                                    hg_id=i)
            outputs.append(head_to_loss)

        model = Model(inputs=input_layer, outputs=outputs)
        rms = RMSprop(lr=5e-4)
        model.compile(optimizer=rms,
                      loss=mean_squared_error,
                      metrics=["accuracy"])

        return model

    def __hourglass_module(self, bottom, bottleneck, hg_id):
        # Create left features: f1, f2, f4, f8
        left_features = self.__create_left_half_blocks(bottom, bottleneck, hg_id)

        # Create right features and connect them with left features
        rf1 = self.__create_right_half_blocks(left_features, bottleneck, hg_id)

        # Add 1x1 conv with two heads:
        # - head_next_stage is sent to the next stage
        # - head_parts is used for intermediate supervision
        head_next_stage, head_parts = self.__create_heads(bottom, rf1, hg_id)

        return head_next_stage, head_parts

    def __create_front_module(self, input_layer, bottleneck):
        """
        Creates the front module of the network
        
        :param input_layer: the input layer of the network 
        :param bottleneck: the type of bottleneck
        :return: the front module layers of the network
        
        Note that the front module has the following structure:
         -> input to 1/4 resolution
         - 1 7x7 conv + max pooling
         - 3 residual block
        """

        x = Conv2D(filters=64,
                   kernel_size=(7, 7),
                   strides=(2, 2),
                   padding='same',
                   activation='relu',
                   name='front_conv_1x1_x1')(input_layer)

        x = BatchNormalization()(x)

        x = bottleneck(x, self.__num_channels // 2, 'front_residual_x1')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = bottleneck(x, self.__num_channels // 2, 'front_residual_x2')
        x = bottleneck(x, self.__num_channels, 'front_residual_x3')

        return x

    def __create_left_half_blocks(self, bottom, bottleneck, hg_layer):
        """
        Creates the left half blocks for hourglass module

        :param bottom:
        :param bottleneck:
        :param hg_layer:
        :return:

        Thanks to these block, at each max pooling step the network branches off and applies
         more convolutions at the original pre-pooled resolution.

        Note the following layer-resolution correspondence:
        - f1: 1
        - f2: 1/2
        - f4: 1/4
        - f8: 1/8
        """

        hg_name = 'hg' + str(hg_layer)

        f1 = bottleneck(bottom, self.__num_channels, hg_name + '_l1')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

        f2 = bottleneck(x, self.__num_channels, hg_name + '_l2')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

        f4 = bottleneck(x, self.__num_channels, hg_name + '_l4')
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

        f8 = bottleneck(x, self.__num_channels, hg_name + '_l8')

        return f1, f2, f4, f8

    def __connect_left_to_right(self, left, right, bottleneck, name):
        """
        Connect the left block to the right ones

        :param left: connect left feature to right feature
        :param name: layer name
        :return: the connection layer between a left and right block
        
        Note that:
        - left  -> 1 bottleneck
        - right -> upsampling
        - Add   -> left + right
        """

        x_left = bottleneck(bottom=left,
                            block_name=name + '_connect')

        x_right = UpSampling2D()(right)

        add = Add()([x_left, x_right])

        out = bottleneck(bottom=add,
                         block_name=name + '_connect_conv')

        return out

    def __bottom_layer(self, lf8, bottleneck, hg_id):
        """
        Create the lowest resolution blocks (3 bottleneck blocks + Add)
        
        :param lf8: 
        :param bottleneck: a bottleneck function
        :param hg_id: the base id code of the new layers
        :return: a bottom layer for the hourglass module
        """

        lf8_connect = bottleneck(lf8, self.__num_channels, str(hg_id) + "_lf8")

        x = bottleneck(lf8, self.__num_channels, str(hg_id) + "_lf8_x1")
        x = bottleneck(x, self.__num_channels, str(hg_id) + "_lf8_x2")
        x = bottleneck(x, self.__num_channels, str(hg_id) + "_lf8_x3")

        rf8 = Add()([x, lf8_connect])

        return rf8

    def __create_right_half_blocks(self, left_features, bottleneck, hg_layer):
        """
        Creates the right half blocks of the network

        :param left_features:
        :param bottleneck:
        :param hg_layer:
        :return:

        Convolutional and max pooling layers are used to process features down to a very low resolution.
        After reaching the lowest resolution, the network begins the top-down sequence of upsampling and
         combination of features across scales.
        """

        lf1, lf2, lf4, lf8 = left_features

        rf8 = self.__bottom_layer(lf8=lf8,
                                  bottleneck=bottleneck,
                                  hg_id=hg_layer)

        rf4 = self.__connect_left_to_right(left=lf4,
                                           right=rf8,
                                           bottleneck=bottleneck,
                                           name='hg' + str(hg_layer) + '_rf4')

        rf2 = self.__connect_left_to_right(left=lf2,
                                           right=rf4,
                                           bottleneck=bottleneck,
                                           name='hg' + str(hg_layer) + '_rf2')

        rf1 = self.__connect_left_to_right(left=lf1,
                                           right=rf2,
                                           bottleneck=bottleneck,
                                           name='hg' + str(hg_layer) + '_rf1')

        return rf1

    def __create_heads(self, pre_layer_features, rf1, hg_id):
        """
        Creates two networks heads, one head will go to the next stage, one to the intermediate features

        :param pre_layer_features:
        :param rf1:
        :param hg_id:
        :return:
        """

        head = Conv2D(filter=self.__num_channels,
                      kernel_size=(1, 1),
                      activation='relu',
                      padding='same',
                      name=str(hg_id) + '_conv_1x1_x1')(rf1)

        head = BatchNormalization()(head)

        # For head as intermediate supervision, use 'linear' activation
        head_parts = Conv2D(self.__num_classes,
                            kernel_size=(1, 1),
                            activation='linear',
                            padding='same',
                            name=str(hg_id) + '_conv_1x1_parts')(head)

        head = Conv2D(self.__num_channels,
                      kernel_size=(1, 1),
                      activation='linear',
                      padding='same',
                      name=str(hg_id) + '_conv_1x1_x2')(head)

        head_m = Conv2D(self.__num_channels,
                        kernel_size=(1, 1),
                        activation='linear',
                        padding='same',
                        name=str(hg_id) + '_conv_1x1_x3')(head_parts)

        head_next_stage = Add()([head,
                                 head_m,
                                 pre_layer_features])

        return head_next_stage, head_parts

    def __bottleneck_block(self, bottom, block_name):

        # Skip layer
        if K.int_shape(bottom)[-1] == self.__num_channels:
            skip = bottom
        else:
            skip = Conv2D(self.__num_channels,
                          kernel_size=(1, 1),
                          activation='relu',
                          padding='same',
                          name=block_name + 'skip')(bottom)

        # Residual: 3 conv blocks as [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
        x = Conv2D(self.__num_channels / 2,
                   kernel_size=(1, 1),
                   activation='relu',
                   padding='same',
                   name=block_name + '_conv_1x1_x1')(bottom)

        x = BatchNormalization()(x)

        x = Conv2D(self.__num_channels / 2,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   name=block_name + '_conv_3x3_x2')(x)

        x = BatchNormalization()(x)

        x = Conv2D(self.__num_channels,
                   kernel_size=(1, 1),
                   activation='relu',
                   padding='same',
                   name=block_name + '_conv_1x1_x3')(x)

        x = BatchNormalization()(x)

        x = Add(name=block_name + '_residual')([skip, x])

        return x

    def __bottleneck_mobile(self, bottom, block_name):
        # Skip layer
        if K.int_shape(bottom)[-1] == self.__num_channels:
            skip = bottom
        else:
            skip = SeparableConv2D(self.__num_channels,
                                   kernel_size=(1, 1),
                                   activation='relu',
                                   padding='same',
                                   name=block_name + 'skip')(bottom)

        # Residual: 3 conv blocks as [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
        x = SeparableConv2D(self.__num_channels / 2,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            name=block_name + '_conv_1x1_x1')(bottom)

        x = BatchNormalization()(x)

        x = SeparableConv2D(self.__num_channels / 2,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same',
                            name=block_name + '_conv_3x3_x2')(x)

        x = BatchNormalization()(x)

        x = SeparableConv2D(self.__num_channels,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            name=block_name + '_conv_1x1_x3')(x)

        x = BatchNormalization()(x)

        x = Add(name=block_name + '_residual')([skip, x])

        return x

    @staticmethod
    def euclidean_loss(x, y):
        return K.sqrt(K.sum(K.square(x - y)))

    class EvalCallBack(keras.callbacks.Callback):

        def __init__(self, folder_path, in_res, out_res):
            super().__init__()

            self.folder_path = folder_path
            self.in_res = in_res
            self.out_res = out_res

        def get_folder_path(self):
            return self.folder_path

        def run_eval(self, epoch):
            validation_data = MPIIDataGen("../../data/mpii/mpii_annotations.json",
                                          "../../data/mpii/images",
                                          in_res=self.in_res,
                                          out_res=self.out_res,
                                          is_train=False)

            total_suc, total_fail = 0, 0
            threshold = 0.5

            count = 0
            batch_size = 8
            for img, gth_map, meta in validation_data.generator(batch_size,
                                                                8,
                                                                sigma=2,
                                                                is_shuffle=False,
                                                                with_meta=True):

                count += batch_size
                if count > validation_data.get_dataset_size():
                    break

                out = self.model.predict(img)

                suc, bad = cal_heatmap_acc(out[-1], meta, threshold)

                total_suc += suc
                total_fail += bad

            acc = total_suc * 1.0 / (total_fail + total_suc)

            print('Eval Accuracy ', acc, '@ Epoch ', epoch)

            with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
                xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

        def on_epoch_end(self, epoch, logs=None):
            # This is a walk around to solve model.save() issue
            # in which large network can't be saved due to size.

            # save model to json
            if epoch == 0:
                jsonfile = os.path.join(self.folder_path, "net_arch.json")
                with open(jsonfile, 'w') as f:
                    f.write(self.model.to_json())

            # save weights
            model_name = os.path.join(self.folder_path, "weights_epoch" + str(epoch) + ".h5")
            self.model.save_weights(model_name)

            print("Saving model to ", model_name)

            self.run_eval(epoch)
