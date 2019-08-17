from networks.classes import Model

import pandas as pd


class ModelFRCNN(Model):
    def __init__(self):
        pass

    def _build_dataset(self):
        pass

    def _restore_weights(self, experiment_path):
        pass

    def _compile_model(self):
        pass

    def _setup_callbacks(self):
        """
        Sets up the callbacks for the training of the model.
        """
        pass

    def train(self):
        """
        Compiles and trains the model for the specified number of epochs.
        """
        pass

    def plot_model(self):
        """
        Plots the model in png format.
        """
        pass

    def display_summary(self):
        """
        Displays the architecture of the model.
        """
        pass

    def evaluate(self) -> any:
        """
        Evaluates the model returning some key performance indicators.
        """
        pass

    def predict(self):
        """
        Performs a prediction using the model.
        """
        pass
