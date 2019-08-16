import os

import numpy as np
import pandas as pd

from scripts.test_bounding_boxes.functions.visualization import visualize_bounding_boxes


def main(image_id: str = 'umgy006-026'):
    """
    Test an annotation drawing the corresponding bounding boxes on an image
    :param image_id: the base name of the image to be tested (without file extension)
    """

    # Set up all the paths to the images
    path_to_dataset = os.path.join('datasets', 'kaggle')
    path_to_train_images = os.path.join(path_to_dataset, 'training', 'backup')

    print('\n---------------------------------------------------------------')
    print('                   TEST OF BOUNDING BOXES                      ')
    print('---------------------------------------------------------------\n')

    print('\nTesting bounding boxes for image {}'.format(image_id))

    print('\nRetrieving annotations from image_labels_map.csv...')

    mapping = pd.read_csv(os.path.join(path_to_dataset, 'image_labels_map.csv'))
    mapping.set_index('image_id', inplace=True)

    labels = mapping.loc[image_id, 'labels']

    if not labels:
        raise Exception('Image {} has no characters hence no labels'.format(image_id))
    else:
        labels = np.array(labels.split(" ")).reshape(-1, 5)

    visualize_bounding_boxes(path_to_train_images, image_id, labels)


if __name__ == '__main__':
    main()
