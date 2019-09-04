import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib import font_manager
from matplotlib.patches import Rectangle


class BboxVisualizer:

    def __init__(self, path_to_images: str):
        self.__path_to_images = path_to_images

    def visualize_bboxes(self, image_id: str, labels: []):
        """
        Takes an image_id and return an image with bounding boxes around each character.

        :param image_id: an image base name (i.e. without the file extension)
        :param labels: the labels of the image to be tested
        """

        plt = self.__display_image(os.path.join(self.__path_to_images, image_id + '.jpg'), show=False)
        ax = plt.gca()

        for label in labels:
            ax = self.draw_box_and_text(ax, label)

        plt.show()

    @staticmethod
    def __display_image(path_to_image: str, show: bool = True):
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

    @staticmethod
    def __get_font():
        """
        Sets up a font capable of rendering the characters
        """
        path = os.path.join(os.getcwd(), 'scripts', 'assets', 'fonts', 'NotoSansCJKjp-Regular.otf')
        return font_manager.FontProperties(fname=path)

    def draw_box_and_text(self, ax, label):
        codepoint, x, y, w, h = label
        x, y, w, h = int(x), int(y), int(w), int(h)

        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        ax.text(x + w + 25, y + (h / 2) + 20, self.__unicode_to_character(codepoint),
                fontproperties=self.__get_font(),
                color="r",
                size=16)

        return ax

    @staticmethod
    def __unicode_to_character(unicode):
        """
        Converts a unicode to a character.
        :param unicode: a unicode string
        :return: a the corresponding unicode character
        """

        # Set the path to the list of classes (i.e. characters)
        path_to_classes = os.path.join('datasets', 'kaggle', 'unicode_translation.csv')

        # Create a codepoint to character map
        unicode_map = {codepoint: char for codepoint, char in pd.read_csv(path_to_classes).values}

        return unicode_map[unicode]
