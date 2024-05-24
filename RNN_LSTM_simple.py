"""
File containing a RNN model, with a simple architecture

Author: Xuan Zheng (1090641)
"""

import numpy as np
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.initializers import GlorotNormal
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.src.regularizers import l2
from sklearn.model_selection import train_test_split

from utils import *
from dataset_analysis import preprocess_data

################################################################
# Hyper-parameters and settings
################################################################

# Settings
domain_1_path = 'domain1_train.json'
domain_2_path = 'domain2_train.json'
model_output_path = 'rnn.model.simple'
verbosity = 1       # Output information for each step
evaluate = 1        # Evaluate model after training
rand_state = 42

# Hyperparameters
undersample_rate = 0.2
epochs = 5
batch_size = 64
learning_rate = 0.001
grad_clip_norm = 1.0
es_patience = 3

################################################################
# Load and preprocess datasets
################################################################

# Read in and preprocess data with custom function
X_combined, y_combined, max_sequence_length = preprocess_data(domain_1_path, domain_2_path, undersample_rate=0.2)

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
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=rand_state)

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=verbosity, restore_best_weights=True)

# Work out vocab size for input_dim
vocab_size = len(np.unique(np.concatenate(X_combined)))

# Define RNN model + compile
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_sequence_length,
                    embeddings_initializer=GlorotNormal()))
model.add(Dropout(0.4))
model.add(LSTM(64, activation='tanh', kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid', bias_initializer='zeros', kernel_regularizer=l2(0.01)))
model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=grad_clip_norm),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train with early stopping and track with custom callback
steps_per_epoch = len(X_train) // batch_size
rnn_fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, RNNCustomCallback()], verbose=verbosity)

################################################################
# Evaluate and save the model
################################################################

# Save the trained model
model.save(model_output_path)
print(f'Model saved at: {model_output_path}') if verbosity else ()

# Evaluate the model, displaying plots
if evaluate:
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=verbosity)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')

    # Extract training history
    training_loss = RNNCustomCallback().losses
    training_accuracy = RNNCustomCallback().accuracies
    validation_loss = rnn_fit.history['val_loss']
    validation_accuracy = rnn_fit.history['val_accuracy']

    # Plot training loss and accuracy per step
    RNNCustomCallback().plot_metrics()
