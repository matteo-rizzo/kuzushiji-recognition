import os

from scripts.utils.utils import to_file_name, draw_box_and_text, display_image


def visualize_bounding_boxes(path_to_train_images: str, image_id: str, labels: []):
    """
    Takes an image_id and return an image with bounding boxes around each character
    :param path_to_train_images: the path to the images of the training set
    :param labels: the labels of the image to be tested
    :param image_id: an image base name (i.e. without the file extension)
    """

    plt = display_image(os.path.join(path_to_train_images, to_file_name(image_id)), show=False)
    ax = plt.gca()

    for label in labels:
        ax = draw_box_and_text(ax, label)

    plt.show()
