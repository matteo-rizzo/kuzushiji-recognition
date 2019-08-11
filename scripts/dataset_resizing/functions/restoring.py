import os
from shutil import copy

from scripts.utils.utils import to_id, to_file_name


def restore_dataset_from_backup(path_to_images: str, path_to_annotations: str, path_to_backup: str):
    """
    Copies all the images from the backup to the main folder and deletes the old files.

    :param path_to_images: the path to the old images
    :param path_to_annotations: the path to the old annotations
    :param path_to_backup: the path to the backup images
    """

    print('Restoring the dataset from backup...\n'
          'The dataset now counts:\n'
          '* {n_img} images\n'
          '* {n_ann} annotations'
          .format(n_img=len(os.listdir(path_to_images)),
                  n_ann=len(os.listdir(path_to_annotations))))

    # Delete the current images and annotations
    for img_file_name in os.listdir(path_to_images):
        # Delete the image
        os.remove(os.path.join(path_to_images, img_file_name))

        # Delete the corresponding annotation
        path_to_annotation = os.path.join(path_to_annotations,
                                          to_file_name(to_id(img_file_name), 'xml'))

        if os.path.isfile(path_to_annotation):
            os.remove(path_to_annotation)

    # Copy the backup images into the main folder
    [copy(os.path.join(path_to_backup, image), path_to_images) for image in os.listdir(path_to_backup)]

    print('Restoring performed successfully!\n'
          'The dataset now counts:\n'
          '* {n_img} images\n'
          '* {n_ann} annotations'
          .format(n_img=len(os.listdir(path_to_images)),
                  n_ann=len(os.listdir(path_to_annotations))))
