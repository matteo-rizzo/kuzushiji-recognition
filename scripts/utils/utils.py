import os

import matplotlib.pyplot as plt
import pandas as pd
import regex as re
from PIL import Image
from matplotlib import font_manager
from matplotlib.patches import Rectangle


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


def unicode_to_character(unicode):
    """
    Converts a unicode to a character.
    :param unicode: a unicode string
    :return: a the corresponding unicode character
    """

    # Set the path to the list of classes (i.e. characters)
    path_to_classes = os.path.join('datasets', 'kaggle', 'classes.csv')

    # Create a codepoint to character map
    unicode_map = {codepoint: char for codepoint, char in pd.read_csv(path_to_classes).values}

    return unicode_map[unicode]


def get_unicodes(string):
    """
    Gets all unicode character in a string using a regex
    """
    string = str(string)
    return re.findall(r'U[+][\S]*', string)


def get_font():
    """
    Sets up a font capable of rendering the characters
    """
    path = os.path.join(os.getcwd(), 'scripts', 'assets', 'fonts', 'NotoSansCJKjp-Regular.otf')
    return font_manager.FontProperties(fname=path)


def display_image(path_to_image: str, show: bool = True):
    """
    Displays an image.
    :param show: a boolean flag to indicate if the image must be displayed on stdout
    :param path_to_image: the path to the image to be displayed
    :return a plot
    """

    plt.figure(figsize=(15, 15))
    this_img = Image.open(path_to_image)
    plt.imshow(this_img)

    if show:
        plt.show()

    return plt


def draw_box_and_text(ax, label):
    codepoint, x, y, w, h = label
    x, y, w, h = int(x), int(y), int(w), int(h)

    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
    ax.add_patch(rect)

    ax.text(x + w + 25, y + (h / 2) + 20, unicode_to_character(codepoint),
            fontproperties=get_font(),
            color="r",
            size=16)

    return ax
