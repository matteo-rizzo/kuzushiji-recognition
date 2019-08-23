import os

from scripts.dataset_resizing.functions.resizing import resize_dataset
from scripts.dataset_resizing.functions.restoring import restore_dataset_from_backup


def main(restore: bool = True, resize: bool = False, size: int = 100):
    """
    Resizes the dataset to the given size and/or restores it from the backup.

    :param restore: a boolean flag to restore the dataset from its backup
    :param resize: a boolean flag to resize the dataset to a given size
    :param size: the new size of the dataset
    """

    print('\n---------------------------------------------------------------')
    print('                   RESIZING OF THE DATASET                       ')
    print('---------------------------------------------------------------\n')

    user_ok = input('The script is going to run the following tasks:\n'
                    'Restoration of the dataset from backup :    {restore}\n'
                    'Resizing of the dataset to size {size} :    {resize}\n \n'
                    'Confirm? [Y/n]\n'
                    .format(restore=restore,
                            resize=resize,
                            size=size))

    confirmations = ['y', 'Y', 'yes', 'ok']

    if user_ok in confirmations:

        # Set up all the paths to the items of the dataset
        path_to_dataset = os.path.join('datasets', 'kaggle', 'training')
        path_to_images = os.path.join(path_to_dataset, 'images')
        path_to_annotations = os.path.join(path_to_dataset, 'annotations')

        if restore:
            path_to_backup = os.path.join(path_to_dataset, 'backup')
            restore_dataset_from_backup(path_to_images, path_to_annotations, path_to_backup)
            print('---------------------------------------------------------------\n')

        if resize:
            dataset_size = len(os.listdir(path_to_images))

            if size > dataset_size:
                raise ValueError('Cannot augment dataset!\n'
                                 '* Dataset size is : {dataset}\n'
                                 '* Given size is   : {size}'
                                 .format(dataset=dataset_size,
                                         size=size))

            resize_dataset(size, path_to_images, path_to_annotations)
            print('---------------------------------------------------------------\n')
    else:
        print('Script execution aborted.')


if __name__ == '__main__':
    main()
