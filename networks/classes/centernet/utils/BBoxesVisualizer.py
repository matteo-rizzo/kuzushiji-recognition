import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib import font_manager
from matplotlib.patches import Rectangle


class BBoxesVisualizer:

    def __init__(self, path_to_images: str):
        self.__path_to_images = path_to_images

    def visualize_bboxes(self, image_id: str, labels: []):
        """
        Takes an image_id and return an image with bounding boxes around each character.

        :param image_id: an image base name (i.e. without the file extension)
        :param labels: the labels of the image to be tested
        """

        path_to_image = os.path.join(self.__path_to_images, image_id + '.jpg')
        fig = plt.figure(figsize=(15, 15))

        fig.add_subplot(1, 2, 1)
        plt.imshow(Image.open(path_to_image))
        ax = plt.gca()

        for label in labels:
            ax = self.__draw_box_and_text(ax=ax,
                                          label=label,
                                          font_size=18,
                                          text_color='r',
                                          text_position='side',
                                          show_text=False,
                                          show_bbox=True)

        fig.add_subplot(1, 2, 2)
        plt.imshow(Image.open(path_to_image), alpha=0.3)
        ax = plt.gca()
        for label in labels:
            ax = self.__draw_box_and_text(ax=ax,
                                          label=label,
                                          font_size=24,
                                          text_color='b',
                                          text_position='center',
                                          show_text=True,
                                          show_bbox=False)

        plt.show()

    def __draw_box_and_text(self,
                            ax,
                            label,
                            font_size=18,
                            text_color='r',
                            text_position='side',
                            show_text=True,
                            show_bbox=True):

        codepoint, x, y, w, h = label
        x, y, w, h = int(x), int(y), int(w), int(h)

        if show_bbox:
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)

        if show_text:
            if text_position == 'side':
                x = x + w + 25

            y = y + h / 2 + 20

            ax.text(x=x,
                    y=y,
                    s=self.__unicode_to_character(codepoint),
                    fontproperties=self.__get_font(),
                    color=text_color,
                    size=font_size)

        return ax

    @staticmethod
    def __get_font():
        """
        Sets up a font capable of rendering the characters
        """
        path = os.path.join(os.getcwd(), 'scripts', 'assets', 'fonts', 'NotoSansCJKjp-Regular.otf')
        return font_manager.FontProperties(fname=path)

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
