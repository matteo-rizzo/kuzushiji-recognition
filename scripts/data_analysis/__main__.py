import os

from scripts.data_analysis.functions.inspection import inspect_data
from scripts.data_analysis.functions.visualization import visualize_images


def main(inspection: bool = True, visualization: bool = True):
    """
    Runs an analysis on the data based on the given parameters.
    :param inspection: a boolean flag to run the inspection of the data
    :param visualization: a boolean flag to run the visualization of some images
    """

    # Set up all the paths to the images
    path_to_dataset = os.path.join('datasets', 'kaggle')
    path_to_train_images = os.path.join(path_to_dataset, 'training', 'images')
    path_to_test_images = os.path.join(path_to_dataset, 'testing', 'images')

    print('---------------------------------------------------------------')
    print('                         DATA ANALYSIS                         ')
    print('---------------------------------------------------------------\n')

    # Inspect folders and files relative to the data
    if inspection:
        inspect_data(path_to_dataset, path_to_train_images, path_to_test_images)
        print('---------------------------------------------------------------\n')

    # Visualize some images from the dataset
    if visualization:
        visualize_images(path_to_dataset, path_to_train_images)
        print('---------------------------------------------------------------\n')


if __name__ == '__main__':
    main()
