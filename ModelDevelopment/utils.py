import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from preprocessing import preprocess_person  # Import your preprocessing function
from scipy.stats import mode
from sklearn.decomposition import PCA
from typing import Dict

from dataclasses import dataclass


@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str


def load_data_and_labels(subject_id, visit_number, actions):
    # Load preprocessed data
    directory_path = f"../DataCollection/data/EEGdata/{subject_id}/{visit_number}/"
    res = preprocess_person(
        directory_path,
        actions,
        should_visualize=False,
    )
    eeg_feature_combined = []
    accel_data_combined = []
    action_data_combined = []
    for eeg_feature, accel_data, action_data in res:
        eeg_feature_combined.append(eeg_feature)
        accel_data_combined.append(accel_data)
        action_data_combined.append(action_data)

    # Merge all arrays using np.concatenate
    eeg_feature_combined = np.concatenate(eeg_feature_combined, axis=0)
    accel_data_combined = np.concatenate(accel_data_combined, axis=0)
    action_data_combined = np.concatenate(action_data_combined, axis=0)

    # Print the number of epochs in eeg_data
    num_epochs = len(eeg_feature_combined)
    print(f"Number of epochs in EEG data: {num_epochs}")

    # Print the number of entries in action_data
    num_action_entries = len(action_data_combined)
    print(f"Number of action data entries: {num_action_entries}")

    # Extract the number of samples per epoch (from the last dimension of eeg_data)
    # num_samples_per_epoch = eeg_feature.shape[-1]

    # Fs = 256  # Sampling frequency in Hz

    # Generate frequency values for positive frequencies only (assuming real-valued EEG data)

    # frequencies = np.fft.rfftfreq(num_samples_per_epoch, d=1/Fs)

    # Define channels
    # channels_to_use = [0, 1, 2, 3, 4, 5, 6, 7]

    X = eeg_feature_combined
    y = action_data_combined  # Assuming action_data contains "action_value" column with labels 1, 2, 3, 4
    X = X.reshape(X.shape[0] * X.shape[-1], X.shape[1] * X.shape[2])
    y = y.flatten()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y
