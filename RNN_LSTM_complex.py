"""
File containing a RNN model, with a more complex architecture

Author: Xuan Zheng (1090641)
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.initializers import GlorotNormal
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.src.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *

################################################################
# Hyper-parameters and settings
################################################################

# Settings
domain_1_path = 'domain1_train.json'
domain_2_path = 'domain2_train.json'
model_output_path = 'rnn.model.complex'
verbosity = 1   # Output information for each step
evaluate = 1    # Evaluate model after training
rand_state = 42

# Hyperparameters
epochs = 4
batch_size = 64
learning_rate = 0.001
grad_clip_norm = 1.0
es_patience = 3

################################################################
# Load and preprocess datasets
################################################################

# Read in domain1 dataset
data_df1 = read_dataset(domain_1_path)
data_df2 = read_dataset(domain_2_path)
print(f'Domain1 loaded from {domain_1_path}, Domain2 loaded from {domain_2_path}') if verbosity else ()

# Extract features/labels, combine and shuffle.
X1, X2 = [data_df1['text'], data_df2['text']]
y1, y2 = [data_df1['label'], data_df2['label']]

# Pad sequences to a fixed length
max_sequence_length = max(len(sequence) for sequence in np.concatenate([X1, X2]))
print(f'Max sequence length: {max_sequence_length}') if verbosity else ()
X1 = pad_sequences(X1, maxlen=max_sequence_length)
X2 = pad_sequences(X2, maxlen=max_sequence_length)

# Apply SMOTE to domain 2 only
X2_SMOTE, y2_SMOTE = SMOTE(sampling_strategy='auto', random_state=rand_state).fit_resample(X2, y2)
if verbosity:
    print(f"Size of Domain 1 Dataset: {X1.shape[0]} samples")
    print(f"Size of Domain 2 Dataset: {X2.shape[0]} samples, {X2_SMOTE.shape[0]} with SMOTE")

# Combine and shuffle
X_combined, y_combined = shuffle(np.concatenate((X1, X2)), np.concatenate((y1, y2)), random_state=rand_state)

################################################################
# Build and train the RNN model
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
model.add(LSTM(64, activation='tanh', kernel_initializer=GlorotNormal(), return_sequences=True,
               kernel_regularizer=l2(0.01)))
model.add(LSTM(64, activation='tanh', kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu', kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', bias_initializer='zeros'))
model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=grad_clip_norm),
              loss='binary_crossentropy',
              metrics=['accuracy'])
rnn_fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                    callbacks=[early_stopping], verbose=verbosity)

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
    training_loss = rnn_fit.history['loss']
    training_accuracy = rnn_fit.history['accuracy']
    validation_loss = rnn_fit.history['val_loss']
    validation_accuracy = rnn_fit.history['val_accuracy']

    # Determine the number of completed epochs (in case of early stop)
    num_epochs = len(training_loss)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), training_accuracy, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()
