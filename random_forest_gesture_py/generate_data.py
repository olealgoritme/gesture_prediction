# generate_data.py - Generate a sample dataset for gesture recognition
import pandas as pd
import numpy as np

# Generate a sample dataset
np.random.seed(42)
num_samples = 100000

# Create random data for two fingers
data = {
    'x1_start': np.random.randint(0, 1024, num_samples),
    'y1_start': np.random.randint(0, 768, num_samples),
    'x1_end': np.random.randint(0, 1024, num_samples),
    'y1_end': np.random.randint(0, 768, num_samples),
    'pressure1_start': np.random.uniform(0.1, 1.0, num_samples),
    'pressure1_end': np.random.uniform(0.1, 1.0, num_samples),
    'x2_start': np.random.randint(0, 1024, num_samples),
    'y2_start': np.random.randint(0, 768, num_samples),
    'x2_end': np.random.randint(0, 1024, num_samples),
    'y2_end': np.random.randint(0, 768, num_samples),
    'pressure2_start': np.random.uniform(0.1, 1.0, num_samples),
    'pressure2_end': np.random.uniform(0.1, 1.0, num_samples),
    'timestamp1_start': np.cumsum(np.random.uniform(0.01, 0.1, num_samples)),  # cumulative sum to simulate time progression
    'timestamp1_end': np.cumsum(np.random.uniform(0.01, 0.1, num_samples)) + np.random.uniform(0.01, 0.1, num_samples),
    'timestamp2_start': np.cumsum(np.random.uniform(0.01, 0.1, num_samples)),
    'timestamp2_end': np.cumsum(np.random.uniform(0.01, 0.1, num_samples)) + np.random.uniform(0.01, 0.1, num_samples),
    'label': np.random.choice(['intentional', 'accidental'], num_samples)
}

# Calculate combined start and end timestamps
data['timestamp_start'] = np.minimum(data['timestamp1_start'], data['timestamp2_start'])
data['timestamp_end'] = np.maximum(data['timestamp1_end'], data['timestamp2_end'])

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file_path = './data/gesture_data.csv'
df.to_csv(csv_file_path, index=False)

print(f"Generated data and saved to {csv_file_path}")
