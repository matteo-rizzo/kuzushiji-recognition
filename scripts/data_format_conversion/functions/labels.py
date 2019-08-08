import os

import pandas as pd


def write_class_label_map(classes: pd.DataFrame, path: str):
    """
    Writes the class-label mapping to csv file

    :param classes: the content of the image_labels_map.csv file
    :param path: the path where to save the csv file to
    """

    # If the class-label mapping file already exist, delete it
    if os.path.isfile(path):
        print('\nA class-label mapping file already exists at {}, deleting it...'.format(path))
        os.remove(path)

    # Add the labelling to the dataframe
    classes.insert(loc=0, column='label', value=range(0, len(classes.index)))

    # Write the mapping on a CSV
    classes.to_csv(path, header=None, index=None, sep=' ', mode='a')

    print('\nThe class-label mapping file has been written at', path)


def write_labels_txt(labels: pd.DataFrame, path: str):
    """
    Write the labels to txt file.

    :param labels: the labels as unicode strings
    :param path: the path where to save the txt file to
    """

    # If the file containing the labels already exist, delete it
    if os.path.isfile(path):
        print('\nA labels file already exists at {}, deleting it...'.format(path))
        os.remove(path)

    # Write the names of the labels on a txt
    labels.to_csv(path, header=None, index=None, sep=' ', mode='a')

    print('\nThe labels file has been written at', path)


def generate_labels(path_to_classes: str, path_to_dataset: str):
    """
    Associates each character to a numeric label and writes the mapping on a CSV file,
    then writes the required labels.txt file in the main dataset folder.

    :param path_to_classes: the path where the classes are stored (in string format)
    :param path_to_dataset: the path to the folder where the labels must be stored
    """

    print('Generating the labels...')

    path_to_labels = os.path.join(path_to_dataset, 'labels')

    if not os.path.isdir(path_to_labels):
        print('Creating labels folder at {}...'.format(path_to_labels))
        os.makedirs(path_to_labels)

    path_to_csv = os.path.join(path_to_labels, 'class_name_to_number.csv')
    path_to_txt = os.path.join(path_to_labels, 'labels.txt')

    # Read the list of characters into a dataframe
    classes = pd.read_csv(path_to_classes)

    # Write the class-label mapping to csv file
    write_class_label_map(classes, path_to_csv)

    # Write the labels to txt file
    write_labels_txt(pd.DataFrame(classes['Unicode']), path_to_txt)
