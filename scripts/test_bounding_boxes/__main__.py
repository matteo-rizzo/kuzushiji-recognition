import os

from scripts.test_bounding_boxes.functions.visualization import visualize_from_backup, visualize_from_training


def main(image_id: str = '100241706_00005_2'):
    """
    Test an annotation drawing the corresponding bounding boxes on an image
    :param image_id: the base name of the image to be tested (without file extension)
    """

    # Set up all the paths to the images
    path_to_dataset = os.path.join('datasets', 'kaggle')
    path_to_backup_images = os.path.join(path_to_dataset, 'training', 'backup')
    path_to_train_images = os.path.join(path_to_dataset, 'training', 'images')
    path_to_annotations = os.path.join(path_to_dataset, 'training', 'annotations')

    print('\n---------------------------------------------------------------')
    print('                   TEST OF BOUNDING BOXES                      ')
    print('---------------------------------------------------------------\n')

    print('\nTesting bounding boxes for image {}'.format(image_id))

    print('\n---------------------------------------------------------------')

    # Visualize the bounding boxes from the backup data
    visualize_from_backup(path_to_images=path_to_backup_images,
                          path_to_mapping=os.path.join(path_to_dataset, 'image_labels_map.csv'),
                          image_id=image_id)

    print('\n---------------------------------------------------------------')

    # Visualize the bounding boxes from the current dataset
    visualize_from_training(path_to_images=path_to_train_images,
                            path_to_annotations=path_to_annotations,
                            image_id=image_id)

    print('\n---------------------------------------------------------------')


if __name__ == '__main__':
    main()
