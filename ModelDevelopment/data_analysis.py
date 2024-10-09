import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, NMF, PCA
from scipy.signal import find_peaks

# CNN Model
class EEGFeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(EEGFeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv1d(8, 16, 5)  # Input channels: 8, Output channels: 16, Kernel size: 5
        self.conv2 = nn.Conv1d(16, 32, 5)  # Input channels: 16, Output channels: 32, Kernel size: 5
        self.pool = nn.MaxPool1d(2)  # Max pooling with kernel size 2

        # Use a dummy tensor to dynamically calculate the flattened size for the fully connected layer
        self._calculate_flattened_size()

        self.fc1 = nn.Linear(self.flattened_size, 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 1)

    def _calculate_flattened_size(self):
        # Create a dummy input tensor with the same shape as the real input (batch_size=1, channels=8, sequence_length=1024)
        dummy_input = torch.zeros(1, 8, 1024)

        # Pass the dummy input through the conv and pool layers to determine the output size
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output and get its size
        self.flattened_size = x.view(1, -1).size(1)

    def forward(self, x):
        # Pass through conv layers and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the input for the fully connected layer
        x = x.view(-1, self.flattened_size)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load EEG Data
def load_eeg_data(file_path):
    print(f"Loading EEG file: {file_path}")
    data = pd.read_csv(file_path)
    
    # Debugging: Check for any non-numeric columns
    print("Checking for non-numeric columns...")
    non_numeric_cols = data.select_dtypes(exclude=[np.number])
    if not non_numeric_cols.empty:
        print("Non-numeric columns found:")
        print(non_numeric_cols.head())
    
    # Drop any non-numeric columns (e.g., timestamps)
    data = data.select_dtypes(include=[np.number])
    
    return data

# Perform NMF
def perform_nmf(eeg_data, n_components=8):
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    W = model.fit_transform(np.abs(eeg_data))  # Ensure no negative values for NMF
    H = model.components_
    return W, H

# Detect Repeating Features
def detect_repeating_features(activation_matrix, sampling_rate=256):
    summed_features = activation_matrix.sum(axis=0)
    peaks, _ = find_peaks(summed_features, distance=sampling_rate * 0.5, height=50, prominence=20)
    return peaks

# Main Analysis
eeg_file = '../DataCollection/data/1234/1234/1234/eeg_data_raw.csv'
eeg_data = load_eeg_data(eeg_file)

# Reshape data for CNN
batch_size = 10
n_channels = 8

# Compute the sequence length dynamically based on the data size
total_data_points = eeg_data.shape[0] * eeg_data.shape[1]
sequence_length = total_data_points // (batch_size * n_channels)

print(f"Total data points: {total_data_points}")
print(f"Calculated sequence length: {sequence_length}")

# Reshape EEG data
try:
    eeg_data_cnn = eeg_data.values.reshape(batch_size, n_channels, sequence_length)
    print(f"Reshaped data: {eeg_data_cnn.shape}")
except ValueError as e:
    print(f"Error reshaping data: {e}")
    print("Check if there is an issue with the number of data points or columns.")
    exit()

# Convert reshaped data to tensor
try:
    eeg_data_cnn_tensor = torch.tensor(eeg_data_cnn, dtype=torch.float32)
    print(f"EEG Data Tensor shape: {eeg_data_cnn_tensor.shape}")
except ValueError as e:
    print(f"Error converting to tensor: {e}")
    print("Ensure the data is numeric and properly reshaped.")
    exit()

# CNN Feature Extraction
cnn_model = EEGFeatureExtractorCNN()
cnn_features = cnn_model(eeg_data_cnn_tensor)
print("CNN Features Extracted")

# Perform NMF on reshaped data
W, H = perform_nmf(eeg_data_cnn.reshape(-1, n_channels))
peaks = detect_repeating_features(H)

# **UPDATED SECTION**: Plot NMF Components with Peaks
# Plot all NMF components in one figure with subplots
n_components = H.shape[0]
fig, axs = plt.subplots(n_components, 1, figsize=(10, n_components * 3))

for i in range(n_components):  # Iterate through each component (rows of H)
    axs[i].plot(H[i, :], label=f'NMF Component {i+1}')
    peaks, _ = find_peaks(H[i, :], distance=sequence_length // 4)  # Adjust peak detection per component
    axs[i].scatter(peaks, H[i, peaks], color='r', label='Detected Peaks')
    axs[i].set_title(f'NMF Component {i+1} Activation with Peaks')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Activation')
    axs[i].legend()

plt.tight_layout()
plt.show()

# Compare CNN Features with NMF
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(cnn_features.detach().numpy()[0, 0, :], label='CNN Feature 1')
plt.title('CNN Feature 1 Activation')
plt.xlabel('Time')
plt.ylabel('Activation')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(H[0, :], label='NMF Component 1')
plt.scatter(peaks, H[0, peaks], color='r', label='Detected Peaks')
plt.title('NMF Component 1 Activation')
plt.xlabel('Time')
plt.ylabel('Activation')
plt.legend()

plt.tight_layout()
plt.show()

# Multi-Component Analysis (NMF, PCA, ICA)
pca = PCA(n_components=8)
ica = FastICA(n_components=8)
pca_features = pca.fit_transform(eeg_data_cnn.reshape(-1, n_channels))
ica_features = ica.fit_transform(eeg_data_cnn.reshape(-1, n_channels))

# Plot NMF, PCA, and ICA
for i in range(2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(H[i, :], label=f'NMF Component {i+1}')
    plt.title(f'NMF Component {i+1}')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(pca_features[:, i], label=f'PCA Component {i+1}')
    plt.title(f'PCA Component {i+1}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(ica_features[:, i], label=f'ICA Component {i+1}')
    plt.title(f'ICA Component {i+1}')
    plt.legend()

    plt.tight_layout()
    plt.show()
