import tensorflow.keras.backend as K

category_n = 1


def all_loss(y_true, y_pred):
    mask = K.sign(y_true[..., 2 * category_n + 2])
    N = K.sum(mask)
    alpha = 2.
    beta = 4.

    heatmap_true_rate = K.flatten(y_true[..., :category_n])
    heatmap_true = K.flatten(y_true[..., category_n:(2 * category_n)])
    heatmap_pred = K.flatten(y_pred[..., :category_n])

    heatloss = -K.sum(
        heatmap_true *
        ((1 - heatmap_pred) ** alpha) * K.log(heatmap_pred + 1e-6) +
        (1 - heatmap_true) *
        ((1 - heatmap_true_rate) ** beta) * (heatmap_pred ** alpha) * K.log(1 - heatmap_pred + 1e-6))

    offsetloss = K.sum(
        K.abs(y_true[..., 2 * category_n] - y_pred[..., category_n] * mask) +
        K.abs(y_true[..., 2 * category_n + 1] - y_pred[..., category_n + 1] * mask))

    sizeloss = K.sum(
        K.abs(y_true[..., 2 * category_n + 2] - y_pred[..., category_n + 2] * mask) +
        K.abs(y_true[..., 2 * category_n + 3] - y_pred[..., category_n + 3] * mask))

    return (heatloss + 1.0 * offsetloss + 5.0 * sizeloss) / N


def size_loss(y_true, y_pred):
    mask = K.sign(y_true[..., 2 * category_n + 2])
    N = K.sum(mask)
    sizeloss = K.sum(
        K.abs(y_true[..., 2 * category_n + 2] - y_pred[..., category_n + 2] * mask) +
        K.abs(y_true[..., 2 * category_n + 3] - y_pred[..., category_n + 3] * mask))
    return (5 * sizeloss) / N


def offset_loss(y_true, y_pred):
    # Filter offset with height ratio > 0
    mask = K.sign(y_true[..., 2 * category_n + 2])
    # Number of points (objects) with ratio > 0
    N = K.sum(mask)
    # sum = abs(target elements height offset - confidence * mask) + abs(target ele width off - height
    # offset * mask)
    offsetloss = K.sum(
        K.abs(y_true[..., 2 * category_n] - y_pred[..., category_n] * mask) +
        K.abs(y_true[..., 2 * category_n + 1] - y_pred[..., category_n + 1] * mask))
    return offsetloss / N


def heatmap_loss(y_true, y_pred):
    mask = K.sign(y_true[..., 2 * category_n + 2])
    N = K.sum(mask)
    alpha = 2.
    beta = 4.

    # Slice the heatmap
    heatmap_true_rate = K.flatten(y_true[..., :category_n])  # column 0 (gaussian heatmap)
    heatmap_true = K.flatten(y_true[..., category_n:(2 * category_n)])  # column 1 (centers)
    heatmap_pred = K.flatten(y_pred[..., :category_n])  # column 0 (score of predicted centers)
    heatloss = -K.sum(
        heatmap_true *
        ((1 - heatmap_pred) ** alpha) * K.log(heatmap_pred + 1e-6) +
        (1 - heatmap_true) *
        ((1 - heatmap_true_rate) ** beta) * (heatmap_pred ** alpha) * K.log(
            1 - heatmap_pred + 1e-6))
    return heatloss / N
