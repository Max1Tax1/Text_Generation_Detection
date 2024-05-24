"""
File containing utility functions

Author: Xuan Zheng (1090641)
"""

import pandas as pd
import json

from keras.src.callbacks import Callback
import matplotlib.pyplot as plt


def tokens_to_text(tokens):
    """
    Function to convert token indices to string format, for TF-IDF
    """
    return ' '.join(map(str, tokens))


def read_dataset(path):
    """
    Function to read dataset from file
    @:return Pandas Dataframe
    """
    data_list = []
    with open(path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            data_list.append(json_data)
    return pd.DataFrame(data_list)


class RNNCustomCallback(Callback):
    def __init__(self):
        super().__init__()
        self.accuracies = []
        self.losses = []

    def on_batch_end(self, epoch, logs=None):
        # Collect accuracy and loss values at the end of each epoch
        self.losses.append(logs.get('val_loss'))
        self.accuracies.append(logs.get('val_accuracy'))

    def plot_metrics(self):
        # Plot accuracy and loss at the end of training
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.losses)), self.losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Per Batch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.accuracies)), self.accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Per Batch')
        plt.legend()
        plt.show()
