import regex as re
from PIL import Image


def to_file_name(image_name: str, extension: str = 'jpg'):
    """
    Converts the name of an image to the corresponding file name.
    Example: image -> image.jpg
    :param extension: the file extension
    :param image_name: the name of an image
    :return: the file name of the given image
    """
    if ".jpg" not in image_name:
        image_name = image_name + '.' + extension
    return image_name


def to_id(file_name):
    """
    Converts the file name of an image to the corresponding base name.
    Example: image.jpg -> image
    :param file_name: the name of an image file (JPG)
    :return: the base name of the given image file
    """
    if file_name[-4:] == ".jpg":
        file_name = file_name[:-4]
    return file_name


def split_characters(string):
    string = str(string)
    string = (re.findall(r"(?:\S*\s){5}", string))
    return [line[:-1] for line in string]


def get_image_size(path_to_image):
    image = Image.open(path_to_image)
    return image.size
