import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample data
trial = 1
eeg_data_path = f"../DataCollection/data/103/1/1/"
data = pd.read_csv(eeg_data_path +"eeg_data_raw.csv")  # replace with your data
labels = pd.read_csv(eeg_data_path +"action_data.csv")

data['timestamp'] = pd.to_datetime(data['timestamp'])
labels['timestamp'] = pd.to_datetime(labels['timestamp'])

merged_data = pd.merge_asof(data, labels, on='timestamp', direction='backward')

timestamps = merged_data['timestamp']       # Assuming a 'timestamp' column exists
time_series_columns = merged_data.columns.difference(['timestamp', 'action_value'])  # Time-domain columns

# Apply FFT
fft_data = merged_data[time_series_columns].apply(lambda col: np.abs(fft(col.values)), axis=0)

# Whiten data
scaler = StandardScaler()
whitened_fft_data = scaler.fit_transform(fft_data)


# Initialize t-SNE with parameters
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)

# Apply t-SNE on the whitened FFT data
tsne_embedding = tsne.fit_transform(whitened_fft_data)

# Convert timestamps to numerical values
timestamp_numeric = pd.to_datetime(timestamps).apply(lambda x: x.timestamp())

# Plotting the t-SNE embedding
plt.figure(figsize=(10, 8))
plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=merged_data['action_value'], cmap='viridis', s=10)
plt.colorbar(label='Timestamp')
plt.title('t-SNE projection of FFT-transformed whitened data colored by action value')
plt.show()