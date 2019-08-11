import os

from scripts.utils.utils import to_file_name, to_id


def resize_dataset(size: int, path_to_images: str, path_to_annotations: str = None):
    """
    Resizes the dataset to the given size deleting all the exceeding items
    and corresponding annotations.

    :param size: the new size of the dataset
    :param path_to_images: the path to the images of the dataset
    :param path_to_annotations: the path to the annotations of the dataset
    """

    print('Resizing the dataset from {old_n} to {new_n} items...'.format(old_n=len(os.listdir(path_to_images)),
                                                                         new_n=size))

    # Iterate over all the dataset items (i.e. names of images)
    for i, img_file_name in enumerate(sorted(os.listdir(path_to_images))):

        # Get the base name of the image (without extension)
        img_id = to_id(img_file_name)

        # Remove all the exceeding items
        if i >= size:
            # Delete the image
            os.remove(os.path.join(path_to_images, to_file_name(img_id, 'jpg')))

            # If annotations have been specified
            if path_to_annotations:

                path_to_annotation = os.path.join(path_to_annotations, to_file_name(img_id, 'xml'))

                # Delete the annotation
                if os.path.isfile(path_to_annotation):
                    os.remove(path_to_annotation)

    print('Resizing Complete!')
    print('The dataset now has:')
    print('* {} images'.format(len(os.listdir(path_to_images))))
    if path_to_annotations:
        print('* {} annotations'.format(len(os.listdir(path_to_annotations))))
