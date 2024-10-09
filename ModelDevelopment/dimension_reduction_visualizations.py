import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

# Sample data
eeg_data_path = f"../DataCollection/data/103/1/1/"
data = pd.read_csv(eeg_data_path +"eeg_data_raw.csv")
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

# Convert timestamps to numerical values
timestamp_numeric = pd.to_datetime(timestamps).apply(lambda x: x.timestamp())


#------------------------------------------------------
#dimension reduction functions
#T-SNE
def plotWithTSNE(data, coloring, colored_by):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_embedding = tsne.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=coloring, cmap='viridis', s=10)
    plt.colorbar(label='Timestamp')
    plt.title(f"t-SNE projection of FFT-transformed whitened data colored by {colored_by}")
    plt.show()
    
#UMAP
def plotWithUMAP(data, coloring, colored_by):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    print(embedding.shape)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=coloring, cmap='viridis', s=10)
    plt.colorbar(label='Timestamp')
    plt.title(f"UMAP projection of FFT-transformed whitened data colored by {colored_by}")
    plt.show()
    
#------------------------------------------------------
# To color by action tag, put "merged_data['action_value'] into second argument"
# To color by timestamp, put "timestamp_numeric" into second argument"

plotWithTSNE(whitened_fft_data, merged_data['action_value'], "action value")
# plotWithUMAP(whitened_fft_data, timestamp_numeric, "timestamp")