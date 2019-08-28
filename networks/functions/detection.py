from typing import List

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from networks.classes.CenterNetDataset import CenterNetDataset
from networks.classes.ModelUtilities import ModelUtilities
from networks.functions import losses
from networks.functions.utils import get_bb_boxes


def run_detection(dataset_params,
                  model_params,
                  dataset_avg_size,
                  input_shape,
                  weights_path,
                  logs) -> (List, List):
    """
    Creates and runs a CenterNet to perform the image detection

    :param dataset_params: the parameters related to the dataset
    :param model_params: the parameters related to the network
    :param dataset_avg_size: a ratio predictor
    :param input_shape: the input shape of the images (usually 512x512x3)
    :param weights_path: the path to the saved weights (if present)
    :param logs: the loggers (execution, training and test)
    :return: a couple of list with train and bbox data.

    Train list has the following structure:
        - train_list[0] = path to image
        - train_list[1] = annotations (ann)
        - train_list[2] = recommended height split
        - train_list[3] = recommended width split
        Where annotations is the bbox data:
        - ann[:, 1] = xmin
        - ann[:, 2] = ymin
        - ann[:, 3] = x width
        - ann[:, 4] = y height

    The bbox data consists of a list with the following structure (note that all are non numeric types):
     [<image_path>, <category>, <score>, <ymin>, <xmin>, <ymax>, <xmax>]

    The <category> value is always 0, because it is not the character category but the category of the center.
    """

    # Generate the CenterNet model
    model_utils = ModelUtilities()
    model = model_utils.generate_model(input_shape=input_shape, mode=2)
    model.compile(optimizer=Adam(lr=model_params['learning_rate']),
                  loss=losses.all_loss,
                  metrics=[losses.size_loss,
                           losses.heatmap_loss,
                           losses.offset_loss])

    # Restore the saved weights if required
    if model_params['restore_weights']:
        model_utils.restore_weights(model=model,
                                    logger=logs['execution'],
                                    init_epoch=model_params['initial_epoch'],
                                    weights_folder_path=weights_path)

    # Get labels from dataset and compute the recommended split
    avg_sizes: List[float] = dataset_avg_size.get_dataset_labels()
    train_list = dataset_avg_size.annotate_split_recommend(avg_sizes)

    # Generate the dataset for detection
    dataset_params['batch_size'] = model_params['batch_size']
    dataset_detection = CenterNetDataset(dataset_params)
    x_train, x_val = dataset_detection.generate_dataset(train_list)
    detection_ts, detection_ts_size = dataset_detection.get_training_set()
    detection_vs, detection_vs_size = dataset_detection.get_validation_set()

    # Train the model
    if model_params['train']:
        logs['execution'].info('Starting the training procedure for model 2 (CenterNet)...')

        # Set up the callbacks
        callbacks = model_utils.setup_callbacks(weights_log_path=weights_path,
                                                batch_size=model_params['batch_size'])

        # Start the training procedure
        model_utils.train(model=model,
                          logger=logs['training'],
                          init_epoch=model_params['initial_epoch'],
                          epochs=model_params['epochs'],
                          training_set=detection_ts,
                          validation_set=detection_vs,
                          training_steps=int(detection_ts_size // model_params['batch_size']) + 1,
                          validation_steps=int(detection_vs_size // model_params['batch_size']) + 1,
                          callbacks=callbacks)

        # Evaluate the model
        metrics = model_utils.evaluate(model=model,
                                       logger=logs['test'],
                                       evaluation_set=detection_vs,
                                       evaluation_steps=int(
                                           detection_vs_size // model_params['batch_size'] + 1))

        logs['test'].info('Evaluation metrics:\n'
                          'all_loss     : {}\n'
                          'size_loss    : {}\n'
                          'heatmap_loss : {}\n'
                          'offset_loss  : {}'
                          .format(metrics[0],
                                  metrics[1],
                                  metrics[2],
                                  metrics[3]))

    # Utility function for resizing
    def resize_fn(path):
        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, (input_shape[1], input_shape[0]))

        return image_resized / 255

    # Prepare a test dataset from the validation set taking its first 10 values
    test_path_list = [ann[0] for ann in x_val[:10]]
    test_ds = tf.data.Dataset.from_tensor_slices(test_path_list) \
        .map(resize_fn,
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(1) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Perform the prediction on the newly created dataset
    detected_predictions = model_utils.predict(model, logs['test'], test_ds, steps=10)

    # get_bb_boxes returns a dict of {image_path: [category,score,ymin,xmin,ymax,xmax]}.
    # Category is always 0. It is not the character category. It's the center category.
    return train_list, get_bb_boxes(detected_predictions, x_val[:10], print=False)
