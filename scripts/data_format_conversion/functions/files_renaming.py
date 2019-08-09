import os


def normalize_files(path_to_files: str):
    """
    Normalizes all the files at the given path by replacing '-' with '_'.

    :param path_to_files: the path to the files to be normalized
    """

    for filename in os.listdir(path_to_files):
        if '-' in filename:
            # Set a new name for the file
            dst = filename.replace('-', '_')

            # Set the source of the file
            src = os.path.join(path_to_files, filename)

            # Set the destination of the file (with the new name)
            dst = os.path.join(path_to_files, dst)

            # Rename the file
            os.rename(src, dst)


def rename_dataset_files(path_to_train_images: str,
                         path_to_test_images: str,
                         path_to_train_annotations: str):
    """
    Renames all the files of the dataset normalizing bad characters.

    :param path_to_train_images: the path to the images used for training
    :param path_to_test_images: the path to the images used for testing
    :param path_to_train_annotations: the path to the training annotations
    """

    print('Renaming files at {}...'.format(path_to_train_images))
    normalize_files(path_to_files=path_to_train_images)

    print('\nRenaming files at {}...'.format(path_to_test_images))
    normalize_files(path_to_files=path_to_test_images)

    if os.listdir(path_to_train_annotations):
        print('\nRenaming files at {}...'.format(path_to_train_annotations))
        normalize_files(path_to_files=path_to_train_annotations)

    print('\nAll files have been renamed successfully!')
