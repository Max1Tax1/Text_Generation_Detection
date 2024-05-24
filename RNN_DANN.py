"""
File containing a RNN model with an integrated DANN architecture

Author: Xuan Zheng (1090641)
"""

import random

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping
from keras.initializers import GlorotNormal
from keras.layers import Embedding, Dense, Dropout, LSTM, Lambda, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.src.regularizers import l2
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

################################################################
# Hyper-parameters and settings
################################################################

# Settings
from utils import read_dataset

domain_1_path = 'domain1_train.json'
domain_2_path = 'domain2_train.json'
model_output_path = 'rnn.model.dann'
verbosity = 1  # Output information for each step
evaluate = 1  # Evaluate model after training
rand_state = 42

# Hyperparameters
undersample_rate = 0.2
epochs = 10  # TODO: 5 EPOCHS LOOKS GOOD
batch_size = 64
learning_rate = 0.001
grad_clip_norm = 1.0
task_loss = 'binary_crossentropy'
domain_loss = 'binary_crossentropy'
lambda_task = 0.5  # Weight for the task loss
lambda_domain = 1  # Weight for the domain adversarial loss
es_patience = 3

################################################################
# Load and preprocess datasets
################################################################

# Read in and preprocess data with custom function for DANN
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

# Undersampling by removing a percentage of the majority "model"s
remove_percentage = undersample_rate
models_to_remove = [0.0, 1.0, 2.0, 3.0, 6.0]
indices_to_remove = []
for index, row in data_df2.iterrows():
    if row['model'] in models_to_remove and random.random() < remove_percentage:
        indices_to_remove.append(index)
undersampled_data_df2 = data_df2.drop(indices_to_remove)
X2, y2 = [undersampled_data_df2['text'], undersampled_data_df2['label']]

# Pad sequences to a fixed length
X1 = pad_sequences(X1, maxlen=max_sequence_length)
X2 = pad_sequences(X2, maxlen=max_sequence_length)

# Apply SMOTE oversampling
X2, y2 = SMOTE(sampling_strategy='auto', random_state=rand_state).fit_resample(X2, y2)
if verbosity:
    print(f"Size of Domain 1 Dataset: {X1.shape[0]} samples")
    print(f"Size of Domain 2 Dataset: {X2.shape[0]} samples with SMOTE")

# Create domain labels (0 for source domain1, 1 for source domain2)
domain_labels_X1 = np.zeros(len(X1), dtype=np.int64)
domain_labels_X2 = np.ones(len(X2), dtype=np.int64)

# Combine X1 and X2 along with their labels and domain labels
X_combined = np.concatenate((X1, X2))
y_combined = np.concatenate((y1, y2))
domain_labels_combined = np.concatenate((domain_labels_X1, domain_labels_X2))

# Label encoding for y_combined
label_encoder = LabelEncoder()
y_combined = label_encoder.fit_transform(y_combined)

# Shuffle the data
X_combined, y_combined, domain_labels_combined = shuffle(X_combined, y_combined,
                                                         domain_labels_combined, random_state=rand_state)

################################################################
# Build and train the RNN model
# Model Architecture:
#       Embedding layer with Xavier's weight initialization
#       Dropout layer, rate = 0.4
#       LSTM layer with Xavier's weight initialization and L2 regularisation.
#       Dropout layer, rate = 0.4
#       Dense layer, squashes the output to 0 and 1, bias initialised to all 0, and L2 regularisation.
# -> Adam optimiser with gradient clipping
################################################################

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, domain_labels_train, domain_labels_test = train_test_split(
    X_combined, y_combined, domain_labels_combined, test_size=0.2, random_state=rand_state)

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=verbosity, restore_best_weights=True)

# Work out vocab size for input_dim len(np.unique(np.concatenate(X_combined)))
vocab_size = len(np.unique(np.concatenate(X_combined)))

# Define RNN Layers
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_length,
                            embeddings_initializer=GlorotNormal())(input_layer)
dropout_1 = Dropout(0.3)(embedding_layer)
lstm_layer1 = LSTM(64, activation='tanh', kernel_initializer=GlorotNormal(), return_sequences=True,
                   kernel_regularizer=l2(0.01))(dropout_1)
lstm_layer2 = LSTM(64, activation='tanh', kernel_initializer=GlorotNormal(),
                   kernel_regularizer=l2(0.01))(lstm_layer1)
dense_layer1 = Dense(64, activation='relu', kernel_initializer=GlorotNormal(),
                     kernel_regularizer=l2(0.01))(lstm_layer2)
dropout_2 = Dropout(0.3)(dense_layer1)

# Domain Classifier Layers
domain_input_layer = Input(shape=(1,))
gradient_reverse = Lambda(lambda x: tf.keras.backend.reverse(x, axes=0))(dropout_2)
dense_layer2 = Dense(64, activation='relu', kernel_initializer=GlorotNormal(),
                     kernel_regularizer=l2(0.01))(gradient_reverse)
domain_classifier = Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01),
                          name='domain_classifier')(dense_layer2)

# Main Classifier Layer
main_classifier = Dense(1, activation='sigmoid', bias_initializer='zeros', kernel_regularizer=l2(0.01),
                        name='main_classifier')(dropout_2)

# Create a combined model with two inputs and two outputs
combined_model = Model(inputs=[input_layer, domain_input_layer], outputs=[main_classifier, domain_classifier])

# Compile the combined model with the separate loss functions and weights
combined_model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=grad_clip_norm),
                       loss={'main_classifier': task_loss, 'domain_classifier': domain_loss},
                       loss_weights={'main_classifier': lambda_task, 'domain_classifier': lambda_domain},
                       metrics=['accuracy'])

# Fit the model with early stopping and track with custom callback
rnn_fit = combined_model.fit(
    [X_train, domain_labels_train], [y_train, domain_labels_train],
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,  # You can adjust the validation split as needed
    callbacks=[early_stopping],
    verbose=verbosity
)

################################################################
# Evaluate and save the model
################################################################

# Save the trained model
combined_model.save(model_output_path)
print(f'Model saved at: {model_output_path}') if verbosity else ()

# Evaluate the model, displaying plots
if evaluate:
    test_loss, [test_task_loss, test_domain_loss], [test_task_accuracy, test_domain_accuracy] = combined_model.evaluate(
        [X_test, domain_labels_test], [y_test, domain_labels_test], verbose=verbosity)
    print(f'Test Task Accuracy: {test_task_accuracy * 100:.2f}%')
    print(f'Test Task Loss: {test_task_loss:.4f}')
    print(f'Test Domain Accuracy: {test_domain_accuracy * 100:.2f}%')
    print(f'Test Domain Loss: {test_domain_loss:.4f}')

    # Extract training history
    training_loss = rnn_fit.history['main_classifier_loss']
    training_task_loss = rnn_fit.history['main_classifier_loss']
    training_domain_loss = rnn_fit.history['domain_classifier_loss']
    training_task_accuracy = rnn_fit.history['main_classifier_accuracy']
    training_domain_accuracy = rnn_fit.history['domain_classifier_accuracy']
