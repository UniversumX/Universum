import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

# Sample data
trial = 1
eeg_data_path = f"../DataCollection/data/103/1/1/eeg_data_raw.csv"
data = pd.read_csv(eeg_data_path)  # replace with your data
timestamps = data['timestamp']       # Assuming a 'timestamp' column exists
time_series_columns = data.columns.difference(['timestamp'])  # Time-domain columns


# Apply FFT
fft_data = data[time_series_columns].apply(lambda col: np.abs(fft(col.values)), axis=0)

# Whiten data
scaler = StandardScaler()
whitened_fft_data = scaler.fit_transform(fft_data)

# Initialize UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(whitened_fft_data)
print(embedding.shape)

# Convert timestamps to numerical values
timestamp_numeric = pd.to_datetime(timestamps).apply(lambda x: x.timestamp())

# Plotting the UMAP embedding
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=timestamp_numeric, cmap='viridis', s=10)
plt.colorbar(label='Timestamp')
plt.title('UMAP projection of FFT-transformed whitened data colored by timestamp')
plt.show()