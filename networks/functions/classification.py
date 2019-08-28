import os

from tensorflow.python.keras.optimizers import Adam

from networks.classes.CenterNetClassificationDataset import ClassifierDataset
from networks.classes.ModelCenterNet import ModelUtilities
from networks.functions.utils import load_crop_characters


def run_classification(dataset_params,
                       model_params,
                       train_list,
                       bbox_predictions,
                       input_shape,
                       weights_path,
                       logs):
    """
    Classifies each character according to the available classes via a CNN

    :param dataset_params: the parameters related to the dataset
    :param model_params: the parameters related to the network
    :param train_list: a train data list predicted at the object detection step
    :param bbox_predictions: the bbox data predicted at the object detection step
    :param input_shape: the input shape of the images (usually 512x512x3)
    :param weights_path: the path to the saved weights (if present)
    :param logs: the loggers (execution, training and test)
    :return: a couple of list with train and bbox data.
    """

    # Generate a model
    model_utils = ModelUtilities()
    model = model_utils.generate_model(input_shape=input_shape,
                                       mode=3)

    # Restore the weights, if required
    if model_params['restore_weights']:
        model_utils.restore_weights(model=model,
                                    logger=logs['execution'],
                                    init_epoch=model_params['initial_epoch'],
                                    weights_folder_path=weights_path)

    # Compile the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=model_params['learning_rate']),
                  metrics=["accuracy"])

    # Generate training set for model 3

    crop_char_path = os.path.join(os.getcwd(), 'datasets', 'char_cropped')

    # NOTE: the following 2 scripts are only run once to generate the images for training.

    # logs['execution'].info('Getting bounding boxes from annotations...')
    # crop_format = annotations_to_bounding_boxes(train_list)

    # logs['execution'].info('Cropping images to characters...')
    # char_train_list = create_crop_characters_train(crop_format, crop_char_path)

    # logs['execution'].info('Cropping done successfully!')

    train_list_3 = load_crop_characters(crop_char_path, mode='train')
    # train_list_3 is a list[(image_path, char_class)]

    # TODO: now create dataset from cropped images (as [image, category])
    # FIXME: below part is not yet completed

    batch_size = int(model_params['batch_size'])
    dataset_params['batch_size'] = batch_size
    dataset_classification = ClassifierDataset(dataset_params)
    x_train, x_val = dataset_classification.generate_dataset(train_list_3)
    classification_ts, classification_ts_size = dataset_classification.get_training_set()
    classification_vs, classification_vs_size = dataset_classification.get_validation_set()

    if model_params['train']:
        callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
                                                batch_size=batch_size)

        model_utils.train(model=model,
                          logger=logs['training'],
                          init_epoch=model_params['initial_epoch'],
                          epochs=model_params['epochs'],
                          training_set=classification_ts,
                          validation_set=classification_vs,
                          training_steps=int(classification_ts_size // batch_size) + 1,
                          validation_steps=int(classification_vs_size // batch_size) + 1,
                          callbacks=callbacks)
