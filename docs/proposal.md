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

All of this datasets are available [here](https://github.com/rois-codh/kmnist). All 3 of them are provided by the Center for Open Data in the Humanities (CODH) and the National Institute of Japanese Literature (NIJL).

There is also a dataset containing the original images from which single characters where extracted in the above datasets:

- Japanese Classics Discarded Data Set. It is available [here](http://codh.rois.ac.jp/char-shape/).

For the purposes of the competition, this last dataset could be more helpful, since part of the task we are required to do is identifying the bounding boxes around each character. On the contrary, the first three datasets listed above doesn't contain any information on bounding boxes.

## Implementation

We plan to study some literature regarding the task OCR specifically for sign languages. 
Many recent articles propose the use of object detection methods for character recognition. We are planning to study the applicability of these algorithms for our task.

Up to now, we selected the following approaches:

- YOLO algorithm for character detection. This is a popular algorithm for real time object detection which uses a full convolutional neural network to label objects in images and also predicts bounding boxes. The original Yolo implementation uses Darknet neural net framework to train. We plan to train an existing implementation of YOLO using Darkflow, which is an implementation of Darknet in tensorflow, to test the performances for our particular task.
- Faster R-CNN. This nets, while slower than YOLO, are reported to achieve better accuracy. An article describing a keras implementation is [this](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a), and a repository with a keras implementation is [this](https://github.com/kbardool/keras-frcnn).
- Ensemble model. Since our task is a difficult one, we are considering the possibility of training different models with different parameters and the combine their predictions in order to achieve better results.

Some references for object detection algorithms applied to text recognition tasks are the following:

- Siba H, Ihab S, Marwa A. March 2019. *Text Detection using Object Recognition Techniques*. COJ Rob Artificial Intel. Available [here](https://crimsonpublishers.com/cojra/pdf/COJRA.000502.pdf).
- Junho J, Hyung Il K, Jae Woong S, Nam Ik C. June 2019. *Handwritten Text Segmentation via End-to-End Learning of Convolutional Neural Networks*. Available [here](https://arxiv.org/abs/1906.05229).
- Nagaoka Y, Miyazaki T, Sugaya Y,  Omachi S. April 2017. *Text Detection by Faster R-CNN with Multiple Region Proposal Networks*, 2017 IAPR International Conference on Document Analysis and Recognition (ICDAR). Available [here](https://ieeexplore.ieee.org/abstract/document/8270290).

## Cloud computing

We plan to take advantage of Google Cloud Platform services for faster training. We already set up an account and requested the use of one GPU for our project.



