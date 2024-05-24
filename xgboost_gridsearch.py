"""
File containing a XGBoost model, with SMOTE oversampling + TF-IDF vectorisation, and hyperparameter tuning through
grid search

Author: Xuan Zheng (1090641)
"""

import joblib
import numpy as np
from utils import *
from scipy.sparse import vstack
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
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
model_output_path = 'xgb.model'
tfidf_output_path = 'xgb_tfidf.pkl'
tfidf_features = 5000
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}
# Define XGBoost hyperparameters for tuning
# param_grid = {
#     'max_depth': [5, 7, 9],
#     'learning_rate': [0.05, 0.1, 0.2],
#     'n_estimators': [50, 100, 150],
# }
param_grid = {
    'max_depth': [9],
    'learning_rate': [0.1],
    'n_estimators': [150],
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

################################################################
# Model training
################################################################

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Fit the grid search object to find the best hyperparameters
model = xgb.XGBClassifier(**xgb_params)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Initialize the best model with the best hyperparameters
best_model = xgb.XGBClassifier(**xgb_params, **best_params)

# Calculate instance weights based on the best model's predictions
best_model.fit(X_train, y_train)
y_train_probs = best_model.predict_proba(X_train)
instance_weights = 1 / np.max(y_train_probs, axis=1)
sample_weights = np.ones(len(y_train))
sample_weights *= instance_weights

# Retrain the best model with instance weights
best_model.fit(X_train, y_train, sample_weight=sample_weights)

################################################################
# Model evaluation and saving
################################################################

# Model evaluation
y_pred = best_model.predict(X_valid)
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
joblib.dump(best_model, model_output_path)
print(f'Model saved at: {model_output_path}')
joblib.dump(tfidf_vectorizer, tfidf_output_path)
print(f'Model saved at: {tfidf_output_path}')
