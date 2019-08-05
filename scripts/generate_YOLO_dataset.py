import os

from scripts.generate_annotations import generate_annotations
from scripts.generate_labels import generate_labels


def main():
    path_to_labels = os.path.join('datasets', 'kaggle', 'training', 'labels.csv')
    generate_labels(path_to_labels)

    path_to_annotations = os.path.join('datasets', 'kaggle', 'training', 'annotations.csv')
    generate_annotations(path_to_annotations)


if __name__ == '__main__':
    main()
