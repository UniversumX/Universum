import torch

# TODO: 
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

from torch.utils.data import Dataset, DataLoader
import mne
import numpy as np
import torch

class EEGAccelDataset(Dataset):
    def __init__(self, edf_file_path, segment_length, transform=None):
        self.edf_file = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
        self.segment_length = segment_length  # length of each data segment (in samples)
        self.transform = transform  # any transformations like normalization

        # Assuming the first 8 channels are EEG and the next 3 are accelerometer data
        self.eeg_data = self.edf_file.get_data(picks=range(8))  # Adjust channel indices as per your data
        self.accel_data = self.edf_file.get_data(picks=range(8, 11))  # Adjust channel indices as per your data

        self.num_segments = self.eeg_data.shape[1] // self.segment_length

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        start = idx * self.segment_length
        end = start + self.segment_length

        eeg_segment = self.eeg_data[:, start:end]
        accel_segment = self.accel_data[:, start:end]

        if self.transform:
            eeg_segment = self.transform(eeg_segment)
            accel_segment = self.transform(accel_segment)

        return torch.from_numpy(eeg_segment).float(), torch.from_numpy(accel_segment).float()


import mne

# Load the EDF file
raw = mne.io.read_raw_edf("dummy_data/dummy_set.edf", preload=True)

# Print information about the file
# Plot the data
raw.plot(duration=5, n_channels=11)
print(raw.info)

import pyedflib

f = pyedflib.EdfReader("dummy_data/dummy_set.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
# Read each channel
for i in range(n):
    print(f.readSignal(i))
