# Neural networks for text detection

The task of text detection is to detect the required text from images/documents. Often, as the need is, you don't want to read the entire document, rather just a piece of information like credit card number, Aadhaar/PAN card number, name, amount and date from bills, etc. Detecting the required text is a tough task but thanks to deep learning, we'll be able to selectively read text from an image.

Text detection or in general object detection has been an area of intensive research accelerated with deep learning. Today, object detection, and in our case, text detection, can be achieved through two approaches.

* Region-Based detectors

* Single Shot detectors

In **Region-Based methods**, the first objective is to find all the regions which have the objects and then pass those regions to a classifier, which gives us the locations of the required objects. So, it is a two-step process.

Firstly, it finds the bounding box and afterwards, the class of it. This approach is considered more accurate but is comparatively slow as compared to the Single Shot approach. Algorithms like [Faster R-CNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) and [R-FCN](https://arxiv.org/abs/1605.06409) take this approach.

**Single Shot detectors**, however, predict both the boundary box and the class at the same time. Being a single step process, it is much faster. However, it must be noted that Single Shot detectors perform badly while detecting smaller objects. [SSD ](https://arxiv.org/abs/1512.02325)and [YOLO ](https://arxiv.org/abs/1506.02640)are Single Shot detectors.

Often, there is a tradeoff between speed and accuracy while choosing the object detector. For example, Faster R-CNN has the highest accuracy, while YOLO is fastest among all. It is very hard to have a fair comparison among different object detectors. There is no straight answer on which model is the best. Besides the detector types, we need to aware of other choices that impact the performance:

- Feature extractors (VGG16, ResNet, Inception, MobileNet).
- Output strides for the extractor.
- Input image resolutions.
- Matching strategy and IoU threshold (how predictions are excluded in calculating loss).
- Non-max suppression IoU threshold.
- Hard example mining ratio (positive v.s. negative anchor ratio).
- The number of proposals or predictions.
- Boundary box encoding.
- Data augmentation.
- Training dataset.
- Use of multi-scale images in training or testing (with cropping).
- Which feature map layer(s) for object detection.
- Localization loss function.
- Deep learning software platform used.
- Training configurations including batch size, input image resize, learning rate, and learning rate decay.