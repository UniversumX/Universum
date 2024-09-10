import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import mne
import numpy as np
import pyedflib


# TODO: 
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels



""" 
class EEGAccelDataset(Dataset):
    def __init__(self, edf_file_path, segment_length, low_freq=None, high_freq=None, apply_ica=False, downsample_rate=None):
        self.edf_file = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
        
        # Preprocessing steps
        if low_freq is not None and high_freq is not None:
            self.bandpass_filter(low_freq, high_freq)
        if apply_ica:
            self.apply_ica()
        if downsample_rate is not None:
            self.downsample(downsample_rate)
        
        self.segment_length = segment_length

        # Extract EEG and accelerometer data
        self.eeg_data = self.edf_file.get_data(picks=range(8))
        self.accel_data = self.edf_file.get_data(picks=range(8, 11))

        self.num_segments = self.eeg_data.shape[1] // self.segment_length

    def bandpass_filter(self, low_freq, high_freq):
        self.edf_file.filter(low_freq, high_freq, fir_design='firwin')

    def apply_ica(self):
        ica = mne.preprocessing.ICA(n_components=0.95, random_state=97, max_iter=800)
        ica.fit(self.edf_file)
        self.edf_file = ica.apply(self.edf_file)

    def downsample(self, downsample_rate):
        self.edf_file.resample(downsample_rate, npad="auto")

    def detrend(self):
        self.eeg_data = signal.detrend(self.eeg_data, axis=-1)

    def normalize(self):
        self.eeg_data = (self.eeg_data - np.mean(self.eeg_data, axis=1, keepdims=True)) / np.std(self.eeg_data, axis=1, keepdims=True)

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        start = idx * self.segment_length
        end = start + self.segment_length

        eeg_segment = self.eeg_data[:, start:end]
        accel_segment = self.accel_data[:, start:end]

        # Apply detrending and normalization
        eeg_segment = signal.detrend(eeg_segment, axis=-1)
        eeg_segment = (eeg_segment - np.mean(eeg_segment, axis=1, keepdims=True)) / np.std(eeg_segment, axis=1, keepdims=True)

        return torch.from_numpy(eeg_segment).float(), torch.from_numpy(accel_segment).float()
"""  

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

"""  
class EEGAccelDataset(Dataset):
    def __init__(self, edf_file_path, segment_length, transform=None, low_freq=1., high_freq=50.):
        self.edf_file = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
        self.edf_file.filter(low_freq, high_freq, fir_design='firwin')  # Bandpass filter
        
        self.segment_length = segment_length
        self.transform = transform

        # Extract EEG and accelerometer data
        self.eeg_data = self.edf_file.get_data(picks=range(8))
        self.accel_data = self.edf_file.get_data(picks=range(8, 11))

        self.eeg_data = (self.eeg_data - np.mean(self.eeg_data, axis=1, keepdims=True)) / np.std(self.eeg_data, axis=1, keepdims=True)  # Normalize EEG data

        self.num_segments = self.eeg_data.shape[1] // self.segment_length

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        start = idx * self.segment_length
        end = start + self.segment_length

        eeg_segment = self.eeg_data[:, start:end]
        accel_segment = self.accel_data[:, start:end]

        if self.transform:  # Apply any additional transformations
            eeg_segment = self.transform(eeg_segment)
            accel_segment = self.transform(accel_segment)

        # Convert to torch tensors
        eeg_segment = torch.from_numpy(eeg_segment).float()
        accel_segment = torch.from_numpy(accel_segment).float()

        return eeg_segment, accel_segment

"""

# Load the EDF file
raw = mne.io.read_raw_edf("dummy_data/dummy_set.edf", preload=True)

# Print information about the file
# Plot the data
raw.plot(duration=5, n_channels=11)
# print(raw.info)

f = pyedflib.EdfReader("dummy_data/dummy_set.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
# Read each channel
for i in range(n):
    print(f.readSignal(i))
