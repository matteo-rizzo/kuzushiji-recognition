import os


def generate_annotations(path_to_annotations):
    if os.path.isfile(path_to_annotations):
        os.remove(path_to_annotations)
    else:
        file = open(path_to_annotations, 'w')
