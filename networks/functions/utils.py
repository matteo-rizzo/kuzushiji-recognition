import numpy as np
from PIL import ImageDraw


def draw_rectangle(box_and_score, img, color):
    number_of_rect = np.minimum(500, len(box_and_score))

    for i in reversed(list(range(number_of_rect))):
        top, left, bottom, right = box_and_score[i, :]

        top = np.floor(top + 0.5).astype('int32')
        left = np.floor(left + 0.5).astype('int32')
        bottom = np.floor(bottom + 0.5).astype('int32')
        right = np.floor(right + 0.5).astype('int32')

        draw = ImageDraw.Draw(img)

        thickness = 4

        if color == "red":
            rect_color = (255, 0, 0)
        elif color == "blue":
            rect_color = (0, 0, 255)
        else:
            rect_color = (0, 0, 0)

        if i == 0:
            thickness = 4
        for j in range(2 * thickness):  # Disegna diversi strati perchè è sottile
            draw.rectangle([left + j, top + j, right - j, bottom - j],
                           outline=rect_color)

    del draw
    return img


def check_iou_score(true_boxes, detected_boxes, iou_thresh):
    iou_all = []
    for detected_box in detected_boxes:
        y1 = np.maximum(detected_box[0], true_boxes[:, 0])
        x1 = np.maximum(detected_box[1], true_boxes[:, 1])
        y2 = np.minimum(detected_box[2], true_boxes[:, 2])
        x2 = np.minimum(detected_box[3], true_boxes[:, 3])

        cross_section = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)
        all_area = (detected_box[2] - detected_box[0]) \
                   * (detected_box[3] - detected_box[1]) \
                   + (true_boxes[:, 2] - true_boxes[:, 0]) \
                   * (true_boxes[:, 3] - true_boxes[:, 1])

        iou = np.max(cross_section / (all_area - cross_section))
        # argmax=np.argmax(cross_section/(all_area-cross_section))
        iou_all.append(iou)

    score = 2 * np.sum(iou_all) / (len(detected_boxes) + len(true_boxes))
    print("score:{}".format(np.round(score, 3)))

    return score
