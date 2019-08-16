# Cognitive services 2018/19 - Project proposal

Members of the team:

- Matteo Rizzo - xxxxxxxx
- Alessandro Zangari - 1207247

## Objective

This project is inspired from a Kaggle competition available at this link: https://www.kaggle.com/c/kuzushiji-recognition/overview.

The objective of the project is building a model able to transcribe pages of handwritten documents written in ancient Kuzushiji into contemporary Japanese characters.
Kuzushiji is a Japanese cursive writing style which has been used in Japan for over a thousand years. Today there are very few fluent readers of Kuzushiji (only 0.01% of modern Japanese natives).

More informations about Kuzushiji are available on the competition page.

## Datasets

The Kaggle competition provides a dataset of images (3881 for training and 4150 for testing), consisting in scanned pages from some Kuzushiji documents. The competition allows to use for external datasets, and some public datasets we may use are the following:

- Kuzushiji-MNIST: 70.000 28x28 images and 10 classes
- Kuzushiji-49: 270.912 character images and 49 classes of characters
- Kuzushiji-Kanji: 140.426 64x64 images of 3832 Kanji characters (very unbalanced)

All of this datasets are available here: https://github.com/rois-codh/kmnist. All 3 of them are provided by the Center for Open Data in the Humanities (CODH) and the National Institute of Japanese Literature (NIJL).

There is also a dataset containing the original images from which single characters where extracted in the above datasets. This datasets is available here:

- Japanese Classics Discarded Data Set: http://codh.rois.ac.jp/char-shape/

For the purposes of the competition, this last dataset could be more helpful, since part of the task we are required to do is identifying the bounding boxes around each character. On the contrary, the first 3 datasets listed above doesn't contain any information on bounding boxes.

## Implementation

We plan to study some literature regarding the task OCR specifically for sign languages. Up to now, we selected the following approaches:

- YOLO algorithm for character detection. This is a popular algorithm for real time object detection which uses a full convolutional neural network to label objects in images and also predicts bounding boxes. The original Yolo implememtation uses Darknet neural net framework to train. We plan to train an existing implementation of YOLO using Darkflow, which is an implementation of Darknet in tensorflow, to test the performances for our particular task.
- Faster R-CNN. This nets, while slower than YOLO, are reported to achieve better accuracy. An article describing a keras implementation is [this](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a), and a repository with a keras implementation is [this](https://github.com/kbardool/keras-frcnn).
- Ensemble model. We are looking into the possibility to combine this two models in order to take the strengths of this two models.

