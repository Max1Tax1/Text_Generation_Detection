"""
File containing a XGBoost model, with SMOTE oversampling + TF-IDF vectorisation

Author: Xuan Zheng (1090641)
"""

import joblib
from scipy.sparse import vstack
import numpy as np
from utils import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

################################################################
# Hyper-parameters and settings
################################################################

domain_1_path = 'domain1_train.json'
domain_2_path = 'domain2_train.json'
model_output_path = 'xgb_SMOTE.model'
tfidf_output_path = 'xgb_SMOTE_tfidf.pkl'
tfidf_features = 5000

# Define XGBoost hyperparameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 100,
}

################################################################
# Load and preprocess datasets
################################################################

# Read in domain1 dataset
data_df1 = read_dataset(domain_1_path)
data_df2 = read_dataset(domain_2_path)

print(f'Domain1 loaded from {domain_1_path}, Domain2 loaded from {domain_2_path}')

# Extract features (text) and labels
X1 = data_df1['text'].apply(tokens_to_text)
y1 = data_df1['label']
X2 = data_df2['text'].apply(tokens_to_text)
y2 = data_df2['label']

# TF-IDF vectorization for domain1
tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_features)
X1_tfidf = tfidf_vectorizer.fit_transform(X1)
X2_tfidf = tfidf_vectorizer.transform(X2)

# Apply SMOTE to domain2 data to oversample the minority class
X2_resampled, y2_resampled = SMOTE(random_state=42).fit_resample(X2_tfidf, y2)

# Concatenate the feature and label arrays
X_combined = vstack([X1_tfidf, X2_resampled])
y_combined = np.concatenate([y1, y2_resampled])

print(f'Datasets pre-processed with SMOTE')
print(f"Size of Domain 1 Dataset: {X1.shape[0]} samples")
print(f"Size of Domain 2 Dataset: {X2.shape[0]} samples")
print(f"Size of Domain 2 + SMOTE: {X2_resampled.shape[0]} samples")

################################################################
# Model training and saving
################################################################

# Initialize and train XGBoost model with resampled data
X_train, X_valid, y_train, y_valid = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(**xgb_params)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')
print("Classification Report:")
print(classification_report(y_valid, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Human', 'Machine'], yticklabels=['Human', 'Machine'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model and TF-IDF vectorizer
model.save_model(model_output_path)
print(f'Model saved at: {model_output_path}')
joblib.dump(tfidf_vectorizer, tfidf_output_path)
print(f'Model saved at: {tfidf_output_path}')
