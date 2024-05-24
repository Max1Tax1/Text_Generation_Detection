"""
File used for preprocessing data, and also provide information of data input

Author: Xuan Zheng (1090641)
"""


import random

import numpy as np
from imblearn.over_sampling import SMOTE
from keras.src.utils import pad_sequences
from sklearn.utils import shuffle

from utils import *


def preprocess_data(domain_1_path, domain_2_path, undersample_rate=0.2, verbosity=1, rand_state=42):

    """
    A function that reads in the dataset and applies hybrid sampling to imbalanced domain 2.
    :param domain_1_path: File path to domain 1
    :param domain_2_path: File path to domain 2
    :param undersample_rate: Percentage of datapoints removed from dataset 2, from majority "model"s
    :param verbosity: allow for dataset analysis (output info)
    :param rand_state: default =42
    :return: (X, y, max_sequence_length); from combined + shuffled domain 1 and domain 2 datapoints.
    """

    # Read in domain1 dataset
    data_df1 = read_dataset(domain_1_path)
    data_df2 = read_dataset(domain_2_path)
    print(f'Domain1 loaded from {domain_1_path}, Domain2 loaded from {domain_2_path}') if verbosity else ()

    # Extract features/labels.
    X1, X2 = [data_df1['text'], data_df2['text']]
    y1, y2 = [data_df1['label'], data_df2['label']]

    # Find max sequence length
    max_sequence_length = max(len(sequence) for sequence in np.concatenate([X1, X2]))
    print(f'Max sequence length: {max_sequence_length}') if verbosity else ()

    # Find size of both domains
    if verbosity:
        print(f"Size of Domain 1 Dataset: {X1.shape[0]} samples")
        print(f"Size of Domain 2 Dataset: {X2.shape[0]} samples")

        # Number of samples under the 2 labels (0 and 1) from domain 1
        label_counts_domain1 = data_df1['label'].value_counts()
        print(f"Domain 1 Label Counts:")
        print(f"Label 0: {label_counts_domain1.get(0, 0)}")
        print(f"Label 1: {label_counts_domain1.get(1, 0)}")

        # Number of samples under the 2 labels (0 and 1) from domain 2
        label_counts_domain2 = data_df2['label'].value_counts()
        print(f"\nDomain 2 Label Counts:")
        print(f"Label 0: {label_counts_domain2.get(0, 0)}")
        print(f"Label 1: {label_counts_domain2.get(1, 0)}")

        # Number of samples under the 7 different "model" tags from domain 2
        model_counts = data_df2['model'].value_counts()
        print(f"\nDomain 2 Model Counts:")
        for model_id, count in sorted(model_counts.items(), key=lambda x: x[0]):
            print(f"Model {model_id}: {count}")

    # Undersampling by removing a percentage the majority "model"s
    remove_percentage = undersample_rate
    models_to_remove = [0.0, 1.0, 2.0, 3.0, 6.0]
    indices_to_remove = []
    for index, row in data_df2.iterrows():
        if row['model'] in models_to_remove and random.random() < remove_percentage:
            indices_to_remove.append(index)
    undersampled_data_df2 = data_df2.drop(indices_to_remove)
    X2, y2 = [undersampled_data_df2['text'], undersampled_data_df2['label']]
    if verbosity:

        # Number of samples under the 2 labels (0 and 1) from domain 2
        label_counts_domain2 = undersampled_data_df2['label'].value_counts()
        print(f"\nDomain 2 Label Counts:")
        print(f"Label 0: {label_counts_domain2.get(0, 0)}")
        print(f"Label 1: {label_counts_domain2.get(1, 0)}")

        # Number of samples under the 7 different "model" tags from domain 2
        model_counts = undersampled_data_df2['model'].value_counts()
        print(f"\nDomain 2 Model Counts:")
        for model_id, count in sorted(model_counts.items(), key=lambda x: x[0]):
            print(f"Model {model_id}: {count}")

    max_sequence_length = max(len(sequence) for sequence in np.concatenate([X1]))

    # Pad sequences to a fixed length
    X1 = pad_sequences(X1, maxlen=max_sequence_length)
    X2 = pad_sequences(X2, maxlen=max_sequence_length)

    # Apply SMOTE oversampling
    X2_SMOTE, y2_SMOTE = SMOTE(sampling_strategy='auto', random_state=rand_state).fit_resample(X2, y2)
    if verbosity:
        print(f"Size of Domain 1 Dataset: {X1.shape[0]} samples")
        print(f"Size of Domain 2 Dataset: {X2.shape[0]} samples, {X2_SMOTE.shape[0]} with SMOTE")

    # Combine and shuffle
    X_combined, y_combined = shuffle(np.concatenate((X1, X2)), np.concatenate((y1, y2)), random_state=rand_state)

    return X_combined, y_combined, max_sequence_length
