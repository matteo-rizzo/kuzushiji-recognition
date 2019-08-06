import os
import pandas as pd


def generate_labels(path_to_classes, path_to_labels):
    """
    Associates each character to a numeric label and writes the mapping on a CSV file.
    :param path_to_classes: the path where the classes are stored (in string format)
    :param path_to_labels: the path to the CSV where the mapping of the labels must be stored
    """

    print('Generating the labels...')

    # If the labels already exist, delete them
    if os.path.isfile(path_to_labels):
        print('A file already exists, deleting it...')
        os.remove(path_to_labels)

    # Read the list of characters into a dataframe
    classes = pd.read_csv(path_to_classes)

    # Add the labelling to the dataframe
    classes.insert(loc=0, column='label', value=range(0, len(classes.index)))

    # Write the mapping on a CSV
    classes.to_csv(path_to_labels, header=None, index=None, sep=' ', mode='a')

    print('The labels file has been written at', path_to_labels)
