import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

pred_out_w, pred_out_h = 128, 128


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
        for j in range(2 * thickness):  # 薄いから何重にか描く
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
        all_area = (detected_box[2] - detected_box[0]) * (detected_box[3] - detected_box[1]) + (
                true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
        iou = np.max(cross_section / (all_area - cross_section))
        # argmax=np.argmax(cross_section/(all_area-cross_section))
        iou_all.append(iou)
    score = 2 * np.sum(iou_all) / (len(detected_boxes) + len(true_boxes))
    print("score:{}".format(np.round(score, 3)))
    return score


def NMS_all(predicts, category_n, score_thresh, iou_thresh):
    y_c = predicts[..., category_n] + np.arange(pred_out_h).reshape(-1, 1)
    x_c = predicts[..., category_n + 1] + np.arange(pred_out_w).reshape(1, -1)
    height = predicts[..., category_n + 2] * pred_out_h
    width = predicts[..., category_n + 3] * pred_out_w

    count = 0
    for category in range(category_n):
        predict = predicts[..., category]
        mask = (predict > score_thresh)
        # print("box_num",np.sum(mask))
        if mask.all == False:
            continue
        box_and_score = NMS(predict[mask], y_c[mask], x_c[mask], height[mask], width[mask], iou_thresh)
        box_and_score = np.insert(box_and_score, 0, category,
                                  axis=1)  # category,score,top,left,bottom,right
        if count == 0:
            box_and_score_all = box_and_score
        else:
            box_and_score_all = np.concatenate((box_and_score_all, box_and_score), axis=0)
        count += 1
    score_sort = np.argsort(box_and_score_all[:, 1])[::-1]
    box_and_score_all = box_and_score_all[score_sort]
    # print(box_and_score_all)

    _, unique_idx = np.unique(box_and_score_all[:, 2], return_index=True)
    # print(unique_idx)
    return box_and_score_all[sorted(unique_idx)]


def visualize_heatmap(image_path: str, r_width: int, r_height: int, heatmap: np.array):
    img = np.array(Image.open(image_path)).resize(new_shape=(r_width, r_height)).convert('RGB')

    gaussian = heatmap[:, :, 0]
    centers = heatmap[:, :, 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 15))
    axes[0].set_axis_off()
    axes[0].imshow(img)
    axes[1].set_axis_off()
    axes[1].imshow(gaussian)
    axes[2].set_axis_off()
    axes[2].imshow(centers)
    plt.show()
