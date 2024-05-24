# train.py - Train a Random Forest classifier on the generated dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your dataset
data = pd.read_csv('./data/gesture_data.csv')

# Feature extraction for two fingers
data['duration1'] = data['timestamp1_end'] - data['timestamp1_start']
data['duration2'] = data['timestamp2_end'] - data['timestamp2_start']
data['velocity1'] = np.sqrt((data['x1_end'] - data['x1_start'])**2 + (data['y1_end'] - data['y1_start'])**2) / data['duration1']
data['velocity2'] = np.sqrt((data['x2_end'] - data['x2_start'])**2 + (data['y2_end'] - data['y2_start'])**2) / data['duration2']

# Handle cases where duration might be zero or very small
data['velocity1'].replace([np.inf, -np.inf], 0, inplace=True)
data['velocity2'].replace([np.inf, -np.inf], 0, inplace=True)

# Selecting features and target
features = data[['x1_start', 'y1_start', 'x1_end', 'y1_end', 'pressure1_start', 'pressure1_end', 'velocity1',
                 'x2_start', 'y2_start', 'x2_end', 'y2_end', 'pressure2_start', 'pressure2_end', 'velocity2',
                 'timestamp_start', 'timestamp_end']]
target = data['label']

# Encode target labels to numeric values
target = target.map({'intentional': 1, 'accidental': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model checkpoint
checkpoint_path = './models/gesture_model.pkl'
joblib.dump(model, checkpoint_path)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n {report}")
print(f"Model saved to {checkpoint_path}")
