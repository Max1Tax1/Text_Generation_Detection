"""
File used for testing saved RNN model, to produce competition submission

Author: Xuan Zheng (1090641)
"""

from keras.src.utils import pad_sequences
from utils import read_dataset
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Load in saved model
model_input_path = 'rnn.model.simple'
model = tf.keras.models.load_model(model_input_path)

# Load the test data
test_data_df = read_dataset('test_set.json')

# Extract the text data from the test data
X_test_text = test_data_df['text']

# Pad test set
max_sequence_length = 1075
X_test = pad_sequences(X_test_text, maxlen=max_sequence_length)

# Run predictions on the test data
y_pred_test = model.predict(X_test)

# Create a histogram to visualize the predicted values
y_pred_test_flat = y_pred_test.flatten()
plt.figure(figsize=(10, 6))
plt.hist(y_pred_test_flat, bins=100, density=True, alpha=0.75, color='b')
plt.xlabel("Predicted Values")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Values")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Apply the threshold to classify your test data
threshold = 0.5
y_pred_test = (y_pred_test > threshold).astype(int)

# Save the predictions to a CSV file
pred_df = pd.DataFrame({'id': test_data_df['id'], 'class': y_pred_test})
pred_df.to_csv('rnn_pred.csv', index=False)
