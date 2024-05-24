# run.py - Runs inference on the finetuned model
import pandas as pd
import numpy as np
import joblib
import os

# Function to load the model with fallback
def load_model():
    try:
        model = joblib.load('./models/finetuned_gesture_model.pkl')
        print("Loaded finetuned_gesture_model.pkl")
    except (FileNotFoundError, Exception) as e:
        model = joblib.load('./models/gesture_model.pkl')
        print("Loaded gesture_model.pkl")
    return model

# Load the model checkpoint with fallback
model = load_model()

# Example new gesture data for inference
new_gesture_data = {
    'x1_start': [400],
    'y1_start': [300],
    'x1_end': [500],
    'y1_end': [350],
    'pressure1_start': [0.8],
    'pressure1_end': [0.6],
    'x2_start': [450],
    'y2_start': [350],
    'x2_end': [550],
    'y2_end': [400],
    'pressure2_start': [0.6],
    'pressure2_end': [0.4],
    'timestamp1_start': [0.05],
    'timestamp1_end': [0.1],
    'timestamp2_start': [0.05],
    'timestamp2_end': [0.1]
}

# Calculate combined start and end timestamps
new_gesture_data['timestamp_start'] = [min(new_gesture_data['timestamp1_start'][0], new_gesture_data['timestamp2_start'][0])]
new_gesture_data['timestamp_end'] = [max(new_gesture_data['timestamp1_end'][0], new_gesture_data['timestamp2_end'][0])]

# Convert to DataFrame
new_gesture_df = pd.DataFrame(new_gesture_data)

# Feature extraction
new_gesture_df['duration1'] = new_gesture_df['timestamp1_end'] - new_gesture_df['timestamp1_start']
new_gesture_df['duration2'] = new_gesture_df['timestamp2_end'] - new_gesture_df['timestamp2_start']
new_gesture_df['velocity1'] = np.sqrt((new_gesture_df['x1_end'] - new_gesture_df['x1_start'])**2 + (new_gesture_df['y1_end'] - new_gesture_df['y1_start'])**2) / new_gesture_df['duration1']
new_gesture_df['velocity2'] = np.sqrt((new_gesture_df['x2_end'] - new_gesture_df['x2_start'])**2 + (new_gesture_df['y2_end'] - new_gesture_df['y2_start'])**2) / new_gesture_df['duration2']

# Handle cases where duration might be zero or very small
new_gesture_df['velocity1'].replace([np.inf, -np.inf], 0, inplace=True)
new_gesture_df['velocity2'].replace([np.inf, -np.inf], 0, inplace=True)

# Selecting features
new_features = new_gesture_df[['x1_start', 'y1_start', 'x1_end', 'y1_end', 'pressure1_start', 'pressure1_end', 'velocity1',
                               'x2_start', 'y2_start', 'x2_end', 'y2_end', 'pressure2_start', 'pressure2_end', 'velocity2',
                               'timestamp_start', 'timestamp_end']]

# Perform inference
prediction = model.predict(new_features)

# Map the prediction back to label
label_map = {1: 'intentional', 0: 'accidental'}
predicted_label = label_map[prediction[0]]

print(f"Predicted Gesture: {predicted_label}")
