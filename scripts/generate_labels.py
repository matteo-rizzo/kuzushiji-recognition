import os
import pandas as pd


def generate_labels(path_to_labels):
    """
    Associates each character to a numeric label and writes the mapping on a CSV file.
    :param path_to_labels: the paths to the CSV which will contain the labels
    """

    # If the labels already exist, delete them
    if os.path.isfile(path_to_labels):
        os.remove(path_to_labels)

    # Read the list of characters into a dataframe
    characters = pd.read_csv(os.path.join('datasets', 'kaggle', 'training', 'characters.csv'),
                             usecols=['Unicode', 'char'])

    # Add the labelling to the dataframe
    characters['label'] = range(0, characters.size)

    # Write the mapping on a CSV
    characters.to_csv(path_to_labels)
