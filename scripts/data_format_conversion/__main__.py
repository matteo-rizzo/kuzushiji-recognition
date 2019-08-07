import os

from scripts.data_format_conversion.functions.labels import generate_labels
from scripts.data_format_conversion.functions.annotations import generate_annotations


def main(labels: bool = False, annotations: bool = True):
    """
    YOLO requires annotations in the format:

    <object-class> <x> <y> <width> <height>.

    Where:
    - <object-class>   : an integer number from 0 to (classes-1) indicating the class of the object
    - <x> <y>          : the coordinates of the center of the rectangle (normalized between 0.0 and 1.0)
    - <width> <height> : float values relative to width and height of the image (normalized between 0.0 and 1.0)

    :param labels: boolean flag to state if labels must be generated
    :param annotations: boolean flag to state if annotations must be generated
    """

    print('---------------------------------------------------------------')
    print('                   DATA FORMAT CONVERSION                      ')
    print('---------------------------------------------------------------\n')

    # Set the base path to the dataset
    path_to_dataset = os.path.join('datasets', 'kaggle')

    # Set the path to the training set
    path_to_training_set = os.path.join(path_to_dataset, 'training')

    # Set the path where the classes are stored (in string format)
    path_to_classes = os.path.join(path_to_dataset, 'classes.csv')

    # --- GENERATE LABELS ---

    if labels:
        # Set the path where the mapping of the labels must be stored
        path_to_labels = os.path.join(path_to_dataset, 'labels.txt')

        # Generate the mapping between labels (integers) and classes (strings)
        generate_labels(path_to_classes, path_to_labels)

        print('---------------------------------------------------------------\n')

    # --- GENERATE ANNOTATIONS ---

    if annotations:
        # Set the path to the images folder
        path_to_images = os.path.join(path_to_training_set, 'images')

        # Set the path to the annotations folder
        path_to_annotations = os.path.join(path_to_training_set, 'annotations')

        # Set the path to the image-labels mapping
        # note that here classes are strings and not integers
        path_to_map = os.path.join(path_to_dataset, 'image_labels_map.csv')

        # Generate a file of annotation for each image
        generate_annotations(path_to_annotations,
                             path_to_images,
                             path_to_map,
                             path_to_classes)

        print('---------------------------------------------------------------\n')


if __name__ == '__main__':
    main()
