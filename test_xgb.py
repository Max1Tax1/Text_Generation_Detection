"""
File used for testing saved XGBoost model, to produce competition submission

Author: Xuan Zheng (1090641)
"""

import joblib
import xgboost as xgb

from utils import *

# Load in saved model and tfidf vectorizer
model_input_path = 'xgb.model'
tfidf_input_path = 'xgb_tfidf.pkl'

model = xgb.XGBClassifier()
model.load_model(model_input_path)
tfidf_vectorizer = joblib.load(tfidf_input_path)

# Load the test data
test_data_df = read_dataset('test_set.json')

# Extract text features and apply tfidf
X_test = test_data_df['text'].apply(tokens_to_text)
X_tfidf_test = tfidf_vectorizer.transform(X_test)

# Use the trained model to make predictions on the test data
y_pred_test = model.predict(X_tfidf_test)

# Save the predictions to a CSV file
pred_df = pd.DataFrame({'id': test_data_df['id'], 'class': y_pred_test})
pred_df.to_csv('test_predictions.csv', index=False)
