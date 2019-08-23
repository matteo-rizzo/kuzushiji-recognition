import os

from PIL import Image


def scale_dataset(path_to_images, scale):
    """
    Scales all the images within the dataset to the given scale
    :param path_to_images: the path to the images of the dataset
    :param scale: the new scale of the images
    """

    print('Scaling the images to {}...'.format(scale))

    # Give a temporary name to the images of the dataset
    for img_file_name in os.listdir(path_to_images):
        os.rename(os.path.join(path_to_images, img_file_name),
                  os.path.join(path_to_images, 'tmp' + img_file_name))

    # Iterate over the images
    for img_file_name in os.listdir(path_to_images):
        # Set the path to the current image
        path_to_img = os.path.join(path_to_images, img_file_name)

        # Open the image
        img = Image.open(path_to_img)

        # Get width and height of the image
        width, height = img.size

        # Perform the resizing
        img.thumbnail((width * scale, height * scale),
                      Image.ANTIALIAS)

        # Save the resized image
        img.save(os.path.join(path_to_images, img_file_name[3:]),
                 format='JPEG',
                 quality=90,
                 optimize=True)

    # Remove all the old images
    for img_file_name in os.listdir(path_to_images):
        if 'tmp' in img_file_name:
            os.remove(os.path.join(path_to_images, img_file_name))
