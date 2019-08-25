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
    heatloss = -K.sum(heatmap_true * ((1 - heatmap_pred) ** alpha) * K.log(heatmap_pred + 1e-6) + (
            1 - heatmap_true) * ((1 - heatmap_true_rate) ** beta) * (heatmap_pred ** alpha) * K.log(
        1 - heatmap_pred + 1e-6))
    offsetloss = K.sum(K.abs(y_true[..., 2 * category_n] - y_pred[..., category_n] * mask) + K.abs(
        y_true[..., 2 * category_n + 1] - y_pred[..., category_n + 1] * mask))
    sizeloss = K.sum(K.abs(y_true[..., 2 * category_n + 2] - y_pred[..., category_n + 2] * mask) + K.abs(
        y_true[..., 2 * category_n + 3] - y_pred[..., category_n + 3] * mask))

    all_loss = (heatloss + 1.0 * offsetloss + 5.0 * sizeloss) / N
    return all_loss


def size_loss(y_true, y_pred):
    mask = K.sign(y_true[..., 2 * category_n + 2])
    N = K.sum(mask)
    sizeloss = K.sum(K.abs(y_true[..., 2 * category_n + 2] - y_pred[..., category_n + 2] * mask) + K.abs(
        y_true[..., 2 * category_n + 3] - y_pred[..., category_n + 3] * mask))
    return (5 * sizeloss) / N


def offset_loss(y_true, y_pred):
    mask = K.sign(y_true[..., 2 * category_n + 2])
    N = K.sum(mask)
    offsetloss = K.sum(K.abs(y_true[..., 2 * category_n] - y_pred[..., category_n] * mask) + K.abs(
        y_true[..., 2 * category_n + 1] - y_pred[..., category_n + 1] * mask))
    return (offsetloss) / N


def heatmap_loss(y_true, y_pred):
    mask = K.sign(y_true[..., 2 * category_n + 2])
    N = K.sum(mask)
    alpha = 2.
    beta = 4.

    heatmap_true_rate = K.flatten(y_true[..., :category_n])
    heatmap_true = K.flatten(y_true[..., category_n:(2 * category_n)])
    heatmap_pred = K.flatten(y_pred[..., :category_n])
    heatloss = -K.sum(heatmap_true * ((1 - heatmap_pred) ** alpha) * K.log(heatmap_pred + 1e-6) + (
            1 - heatmap_true) * ((1 - heatmap_true_rate) ** beta) * (heatmap_pred ** alpha) * K.log(
        1 - heatmap_pred + 1e-6))
    return heatloss / N
