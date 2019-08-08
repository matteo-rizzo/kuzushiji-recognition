import os
import pandas as pd

from scripts.utils.utils import get_unicodes


def inspect_data(path_to_dataset, path_to_train_images, path_to_test_images):
    """
    Inspect the main folders and files relative to the dataset.
    :param path_to_dataset: the path to the kaggle dataset folder
    :param path_to_train_images: the path to the images used for training
    :param path_to_test_images: the path to the images used for testing
    """

    print("Running data inspection...")

    print('\nKaggle dataset folder:')
    print(os.listdir(path_to_dataset))

    print('\nImages for training ({} elements):'.format(len(os.listdir(path_to_train_images))))
    print(os.listdir(path_to_train_images)[:5])

    print('\nImages for testing ({} elements):'.format(len(os.listdir(path_to_test_images))))
    print(os.listdir(path_to_test_images)[:5])

    # Inspect the image-labels mapping
    mapping = pd.read_csv(os.path.join(path_to_dataset, 'image_labels_map.csv'))
    print('\nImage-labels mapping ({} elements):'.format(len(mapping.index)))
    print(mapping.head())
    print('\nBasic info on the mapping:')
    mapping.info()

    # Inspect the classes
    classes = pd.read_csv(os.path.join(path_to_dataset, 'classes.csv'))
    print('\nClasses as characters ({} elements)'.format(len(classes)))
    print(classes.head(10))

    # Concatenate all labels together and get the unicodes
    all_unicodes = get_unicodes(mapping.labels.str.cat(sep=" "))

    # Typecast to set to remove duplicates and then back to list
    print('\nNote that {training} out of {all} unicode characters are present within the training set'
          .format(training=len(list(set(all_unicodes))),
                  all=len(classes)))

    print('\nData inspection has finished!')
