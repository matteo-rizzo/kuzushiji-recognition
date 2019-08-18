import os

from scripts.data_format_conversion.functions.annotations import generate_annotations
from scripts.data_format_conversion.functions.files_renaming import rename_dataset_files
from scripts.data_format_conversion.functions.labels import generate_labels


def main(labels: bool = False,
         annotations: bool = True,
         annotation_format: str = 'frcnn',
         renaming: bool = True):
    """
    YOLO requires annotations in the format:

    <object-class> <x> <y> <width> <height>.

    Where:
    - <object-class>   : an integer number from 0 to (classes-1) indicating the class of the object
    - <x> <y>          : the coordinates of the center of the rectangle (normalized between 0.0 and 1.0)
    - <width> <height> : float values relative to width and height of the image (normalized between 0.0 and 1.0)

    Darkflow requires annotations in the PASCAL VOC XML format.

    :param labels: boolean flag to state if labels must be generated
    :param annotations: boolean flag to state if annotations must be generated
    :param annotation_format: defines the format of the annotations ('YOLOv2' or 'darkflow')
    :param renaming: a boolean flag to rename all the files related to the dataset
    """

    print('\n---------------------------------------------------------------')
    print('                   DATA FORMAT CONVERSION                      ')
    print('---------------------------------------------------------------\n')

    user_ok = input('The script is going to run the following tasks:\n'
                    'Generations of labels:                         {labels}\n'
                    'Generations of annotations in {format} format: {annotations}\n'
                    'Renaming of dataset files:                     {renaming}\n \n'
                    'Confirm? [Y/n]\n'
                    .format(labels=labels,
                            format=annotation_format,
                            annotations=annotations,
                            renaming=renaming))

    confirmations = ['y', 'Y', 'yes', 'ok']

    if user_ok in confirmations:

        # os.chdir('~/repo_project_cs')

        # Set the base path to the dataset
        path_to_dataset = os.path.join('datasets', 'kaggle')

        # Set the path to the training and test set
        path_to_training_set = os.path.join(path_to_dataset, 'training')
        path_to_testing_set = os.path.join(path_to_dataset, 'testing')

        # Set the path to the images folders
        path_to_train_images = os.path.join(path_to_training_set, 'images')
        path_to_test_images = os.path.join(path_to_testing_set, 'images')

        # Set the path where the classes are stored (in string format)
        path_to_classes = os.path.join(path_to_dataset, 'classes.csv')

        # Generate the class name to class number mapping
        if labels:
            # Generate the mapping between labels (integers) and classes (strings)
            generate_labels(path_to_classes, path_to_dataset)

            print('---------------------------------------------------------------\n')

        # Generate the annotations
        if annotations:
            # Set the path to the annotations folder
            path_to_annotations = os.path.join(path_to_training_set, 'annotations')

            # Set the path to the image-labels mapping
            # note that here classes are strings and not integers
            path_to_map = os.path.join(path_to_dataset, 'image_labels_map.csv')

            # Generate a file of annotation for each image
            generate_annotations(path_to_annotations,
                                 path_to_train_images,
                                 path_to_map,
                                 path_to_classes,
                                 annotation_format)

            print('---------------------------------------------------------------\n')

        # Dataset files renaming in order to avoid training errors
        if renaming:
            # Set the path to the annotations folder
            path_to_train_annotations = os.path.join(path_to_dataset, 'training', 'annotations')

            rename_dataset_files(path_to_train_images,
                                 path_to_test_images,
                                 path_to_train_annotations)
            print('---------------------------------------------------------------\n')
    else:
        print('Script execution aborted.')


if __name__ == '__main__':
    main()
