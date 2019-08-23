import logging
import os

import absl.logging
import tensorflow as tf

from networks.classes.CenterNetDataset import CenterNetDataset
from networks.classes.ModelCenterNet import ModelCenterNet
from networks.classes.Logger import Logger
from networks.classes.Params import Params


def main():
    # -- TENSORFLOW BASIC CONFIG ---

    # Enable eager execution
    # tf.compat.v1.enable_eager_execution()
    tf.enable_eager_execution()
    eager_exec_status = str('Yes') if tf.executing_eagerly() else str('No')
    # eager_exec_status = str('Yes') if tf.compat.v1.executing_eagerly else str('No')

    # Set up the log for tensorflow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Remove absl logs
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

    # --- GENERAL PARAMETERS ---

    # Set the path to the configuration folder
    config_path = os.path.join(os.getcwd(), 'networks', 'configuration')

    # Load the general parameters from json file
    centernet_params = Params(os.path.join(config_path, 'params_model_CenterNet.json'))
    dataset_params = centernet_params.dataset
    dataset_params['batch_size'] = centernet_params.network['batch_size']

    # Get the info for the current run
    run_id = centernet_params.model['run_id']
    mode = centernet_params.model['mode']

    # --- LOGGER ---

    log_handler = Logger(run_id)
    log = log_handler.get_logger('execution')

    # Log configuration
    log.info('Software versions:')
    log.info('* Tensorflow version: ' + tf.__version__)
    log.info('* Keras version:      ' + tf.__version__)
    log.info('* Executing eagerly?  ' + eager_exec_status)

    log.info('General parameters:')
    log.info('* Model:               CenterNet - ' + mode)
    log.info('* Training dataset:   ' + dataset_params['train_images_path'])
    log.info('* Test dataset:       ' + dataset_params['test_images_path'] + '\n')

    # Log general and training parameters
    log_handler.log_configuration(run_id, 'CenterNet', implementation=False)

    # --- DATASET ---

    # Import the dataset for training
    log.info('Importing the dataset for training...')
    dataset_all = CenterNetDataset(dataset_params)

    dataset_all.gen_dataset_size_model()

    training_set, t_size = dataset_all.get_training_set()
    validation_set, v_size = dataset_all.get_validation_set()
    test_set, p_size = dataset_all.get_test_set()

    input_shape = (dataset_params['input_width'], dataset_params['input_height'], 3)

    # --- MODEL ---

    log.info('Building the model...')

    # Build the model
    model = ModelCenterNet(run_id=run_id,
                           model_params=centernet_params,
                           mode=centernet_params.model['mode'],
                           training_set=training_set,
                           validation_set=validation_set,
                           test_set=test_set,
                           train_size=t_size,
                           val_size=v_size,
                           test_size=p_size,
                           input_shape=input_shape,
                           log_handler=log_handler)

    # --- TRAINING ---

    # Train the model
    log.info('Starting the training procedure...')

    model.train()
    model.evaluate()

    dataset_all.batch_size = 1

    # Predict average bbox size for training images
    prediction_train = model.predict(dataset=training_set, size=t_size, batch_size=1)

    # Expand dataset adding recommended split size for characters
    dataset_all.gen_dataset_centernet_model(prediction_train)

    # log.info('Testing the model against an image in the training set...')
    # model.predict()

    # --- TEST ---

    # Evaluate training against the given test set


if __name__ == '__main__':
    main()
