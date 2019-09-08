import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class Metrics:

    def __init__(self, category_n=1):
        self.__category_n = category_n

    def all_loss(self, y_true, y_pred):
        mask = K.sign(y_true[..., 2 * self.__category_n + 2])
        N = K.sum(mask)
        alpha = 2.
        beta = 4.

        heat_loss = self.__calculate_heatmap_loss(y_true, y_pred, alpha, beta)
        offset_loss = self.__calculate_offset_loss(y_true, y_pred, mask)
        size_loss = self.__calculate_size_loss(y_true, y_pred, mask)

        return (heat_loss + 1.0 * offset_loss + 5.0 * size_loss) / N

    def __calculate_size_loss(self, y_true, y_pred, mask):
        return K.sum(
            K.abs(y_true[..., 2 * self.__category_n + 2] - y_pred[..., self.__category_n + 2] * mask) +
            K.abs(y_true[..., 2 * self.__category_n + 3] - y_pred[..., self.__category_n + 3] * mask))

    def size_loss(self, y_true, y_pred):
        mask = K.sign(y_true[..., 2 * self.__category_n + 2])
        N = K.sum(mask)

        return (5 * self.__calculate_size_loss(y_true, y_pred, mask)) / N

    def __calculate_offset_loss(self, y_true, y_pred, mask):
        return K.sum(
            K.abs(y_true[..., 2 * self.__category_n] - y_pred[..., self.__category_n] * mask) +
            K.abs(y_true[..., 2 * self.__category_n + 1] - y_pred[..., self.__category_n + 1] * mask))

    def offset_loss(self, y_true, y_pred):
        # Filter offset with height ratio > 0
        mask = K.sign(y_true[..., 2 * self.__category_n + 2])
        # Number of points (objects) with ratio > 0
        N = K.sum(mask)

        return self.__calculate_offset_loss(y_true, y_pred, mask) / N

    def __calculate_heatmap_loss(self, y_true, y_pred, alpha, beta):
        # Column 0 (gaussian heatmap)
        heatmap_true_rate = K.flatten(y_true[..., :self.__category_n])

        # Column 1 (centers)
        heatmap_true = K.flatten(y_true[..., self.__category_n:(2 * self.__category_n)])

        # Column 0 (score of predicted centers)
        heatmap_pred = K.flatten(y_pred[..., :self.__category_n])

        return -K.sum(heatmap_true *
                      ((1 - heatmap_pred) ** alpha) *
                      K.log(heatmap_pred + 1e-6) +
                      (1 - heatmap_true) *
                      ((1 - heatmap_true_rate) ** beta) *
                      (heatmap_pred ** alpha) *
                      K.log(1 - heatmap_pred + 1e-6))

    def heatmap_loss(self, y_true, y_pred):
        mask = K.sign(y_true[..., 2 * self.__category_n + 2])
        N = K.sum(mask)
        alpha = 2.
        beta = 4.

        return self.__calculate_heatmap_loss(y_true, y_pred, alpha, beta) / N
