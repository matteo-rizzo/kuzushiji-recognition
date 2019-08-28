from networks.classes.SizePredictDataset import SizePredictDataset


def run_preprocessing(dataset_params, model_params, input_shape, weights_path,
                      logs) -> SizePredictDataset:
    """
    Creates and runs a CNN which takes an image/page of manuscript as input and predicts the
    average dimensional ratio between the characters and the image itself

    :param dataset_params: the parameters related to the dataset
    :param model_params: the parameters related to the network
    :param input_shape: the input shape of the images (usually 512x512x3)
    :param weights_path: the path to the saved weights (if present)
    :param logs: the loggers (execution, training and test)
    :return: a ratio predictor
    """

    logs['execution'].info('Preprocessing the data...')

    # Build dataset for model 1
    dataset_params['batch_size'] = model_params['batch_size']
    dataset_avg_size = SizePredictDataset(dataset_params)

    dataset_avg_size.generate_dataset()

    size_check_ts, size_check_ts_size = dataset_avg_size.get_training_set()
    size_check_vs, size_check_vs_size = dataset_avg_size.get_validation_set()
    # size_check_ps, size_check_ps_size = dataset_avg_size.get_test_set()
    #
    # # Generate a model
    # model_utils = ModelUtilities()
    # model = model_utils.generate_model(input_shape=input_shape, mode=1)
    # model.compile(loss='mean_squared_error',
    #               optimizer=Adam(lr=model_params['learning_rate']))
    #
    # # Restore the weights, if required
    # if model_params['restore_weights']:
    #     model_utils.restore_weights(model,
    #                                 logs['execution'],
    #                                 model_params['initial_epoch'],
    #                                 weights_path)
    #
    # # Train the model
    # if model_params['train']:
    #     logs['execution'].info('Starting the training procedure for model 1...')
    #
    #     # Set up the callbacks
    #     callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
    #                                             batch_size=model_params['batch_size'])
    #
    #     # Start the training procedure
    #     model_utils.train(model, logs['training'], model_params['initial_epoch'], model_params['epochs'],
    #                       training_set=size_check_ts,
    #                       validation_set=size_check_vs,
    #                       training_steps=int(size_check_ts_size // model_params['batch_size'] + 1),
    #                       validation_steps=int(size_check_vs_size // model_params['batch_size'] + 1),
    #                       callbacks=callbacks)
    #
    #     # Evaluate the model
    #     model_utils.evaluate(model, logger=logs['test'],
    #                          evaluation_set=size_check_vs,
    #                          evaluation_steps=int(size_check_vs_size // model_params['batch_size'] + 1))

    return dataset_avg_size
