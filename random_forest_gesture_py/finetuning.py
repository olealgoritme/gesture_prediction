# finetuning.py - Loads a pre-trained model checkpoint and fine-tunes it with new data.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

# Check if new data file exists
new_data_path = './data/new_gesture_data.csv'
if not os.path.exists(new_data_path):
    print(f"New data file not found: {new_data_path}")
    sys.exit(1)

# Load new dataset
new_data = pd.read_csv(new_data_path)

# Feature extraction for two fingers
new_data['duration1'] = new_data['timestamp1_end'] - new_data['timestamp1_start']
new_data['duration2'] = new_data['timestamp2_end'] - new_data['timestamp2_start']
new_data['velocity1'] = np.sqrt((new_data['x1_end'] - new_data['x1_start'])**2 + (new_data['y1_end'] - new_data['y1_start'])**2) / new_data['duration1']
new_data['velocity2'] = np.sqrt((new_data['x2_end'] - new_data['x2_start'])**2 + (new_data['y2_end'] - new_data['y2_start'])**2) / new_data['duration2']

# Handle cases where duration might be zero or very small
new_data['velocity1'].replace([np.inf, -np.inf], 0, inplace=True)
new_data['velocity2'].replace([np.inf, -np.inf], 0, inplace=True)

# Selecting features and target
new_features = new_data[['x1_start', 'y1_start', 'x1_end', 'y1_end', 'pressure1_start', 'pressure1_end', 'velocity1',
                         'x2_start', 'y2_start', 'x2_end', 'y2_end', 'pressure2_start', 'pressure2_end', 'velocity2',
                         'timestamp_start', 'timestamp_end']]
new_target = new_data['label']

# Encode target labels to numeric values
new_target = new_target.map({'intentional': 1, 'accidental': 0})

# Split the data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_features, new_target, test_size=0.2, random_state=42)

# Load the saved model checkpoint
model = joblib.load('./models/gesture_model.pkl')

# Update the model with new data
model.fit(X_train_new, y_train_new)

# Save the updated model checkpoint
updated_checkpoint_path = './models/finetuned_gesture_model.pkl'
joblib.dump(model, updated_checkpoint_path)

# Make predictions
y_pred_new = model.predict(X_test_new)

# Evaluate the model
accuracy_new = accuracy_score(y_test_new, y_pred_new)
report_new = classification_report(y_test_new, y_pred_new)

print(f"New Accuracy: {accuracy_new}")
print(f"New Classification Report:\n {report_new}")
print(f"Fine-Tuned model saved to {updated_checkpoint_path}")
