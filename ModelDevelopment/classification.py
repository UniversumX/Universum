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


def get_frequency_band_indices(frequencies, band_min, band_max):
    """
    Returns the indices of frequencies that fall within a specified band range.

    Args:
        frequencies (array): Array of frequency values corresponding to the data's frequency dimension.
        band_min (float): Minimum frequency of the band.
        band_max (float): Maximum frequency of the band.

    Returns:
        list: Indices of the frequencies that fall within the specified band range.
    """
    return [i for i, freq in enumerate(frequencies) if band_min <= freq <= band_max]


def extract_features(eeg_data, channels, frequencies):
    """
    Extracts features from the preprocessed EEG data based on mu and beta bands.

    Args:
        eeg_data (numpy array): Preprocessed EEG data of shape (num_epochs, num_channels, num_frequency_bands, num_samples_per_epoch).
        channels (list): List of channel indices to extract features from.
        frequencies (array): Array of frequency values for the third dimension in eeg_data.

    Returns:
        numpy array: Feature matrix of shape (num_epochs, num_features).
    """
    # Get indices for mu (8-12 Hz) and beta (13-30 Hz) bands
    mu_band_indices = get_frequency_band_indices(frequencies, 8, 12)
    beta_band_indices = get_frequency_band_indices(frequencies, 13, 30)
    freq_bands_to_use = mu_band_indices + beta_band_indices

    features = []
    for epoch in eeg_data:  # Iterate over epochs
        epoch_features = []
        for channel_idx in channels:  # Iterate over selected channels
            for band_idx in freq_bands_to_use:  # Iterate over valid frequency bands
                # Compute the mean power across the time samples (amplitude values)
                power = np.mean(
                    np.abs(epoch[channel_idx][band_idx])
                )  # Mean over time samples
                epoch_features.append(power)  # Append feature
        features.append(epoch_features)  # Append features for the epoch
    return np.array(features)


# Example of usage in load_data_and_labels
# Assuming `frequencies` is an array of frequency values corresponding to the third dimension of eeg_data
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

    return X, y

    '''
def load_multiple_trials(subject_id, visit_number, trial_numbers, actions):
    """
    Loads and preprocesses the EEG data from multiple trials and aggregates them.
    
    Args:
        subject_id (str): The subject ID.
        visit_number (int): The visit number.
        trial_numbers (list): A list of trial numbers to load and aggregate.
        actions (dict): Dictionary mapping action names to their corresponding Action objects.
    
    Returns:
        tuple: (X, y) - Aggregated feature matrix and corresponding labels from all trials.
    """
    all_X = []
    all_y = []

    # Loop over all trial numbers
    for trial_number in trial_numbers:
        # Load data from each trial
        X, y = load_data_and_labels(subject_id, visit_number, trial_number, actions)

        # Append the data from this trial to the overall dataset
        all_X.append(X)
        all_y.append(y)
    
    # Concatenate the data from all trials
    X_combined = np.vstack(all_X)  # Stack vertically
    y_combined = np.concatenate(all_y)  # Concatenate labels

    return X_combined, 
    '''


def classify_eeg_data(subject_id, visit_number, actions):
    # Load and preprocess data from multiple trials
    X, y = load_data_and_labels(subject_id, visit_number, actions)

    # Now, split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Organize the training data by action
    action_num = 4
    channel_num = 8

    # Initialize a structure to hold arrays for each channel and each action
    features_by_channel_action = {
        action: [None] * channel_num for action in range(1, action_num + 1)
    }

    # Iterate over actions
    for action in range(1, action_num + 1):
        # Filter data for the current action
        action_data = X_train[y_train == action]

        # Iterate over channels
        for channel in range(channel_num):
            # Extract data for the current channel and store it
            if features_by_channel_action[action][channel] is None:
                features_by_channel_action[action][channel] = action_data[:, channel]
            else:
                features_by_channel_action[action][channel] = np.vstack(
                    (
                        features_by_channel_action[action][channel],
                        action_data[:, channel],
                    )
                )

    # Initialize a dictionary to store GMMs for each action and channel
    gmms_by_channel_action = {
        action: [None] * channel_num for action in range(1, action_num + 1)
    }

    # Initialize a dictionary to store scalers for each action and channel
    scalers_by_channel_action = {
        action: [None] * channel_num for action in range(1, action_num + 1)
    }

    # Train a GMM for each action and channel
    for action in range(1, action_num + 1):
        for channel in range(channel_num):
            # Extract data for the current action and channel
            channel_data = features_by_channel_action[action][channel]

            # Standardize the data for the current action and channel
            scaler = StandardScaler()
            channel_data_scaled = scaler.fit_transform(np.abs(channel_data))

            # Store the scaler
            scalers_by_channel_action[action][channel] = scaler

            # Train a GMM with 1 component for this action and channel
            gmm = GaussianMixture(n_components=1, random_state=42)
            gmm.fit(channel_data_scaled)

            # Store the trained GMM
            gmms_by_channel_action[action][channel] = gmm

    # Test the GMMs
    probabilities = np.zeros((X_test.shape[0], action_num))

    # Standardize and evaluate probabilities for each channel
    for action in range(1, action_num + 1):
        action_probabilities = np.zeros(X_test.shape[0])  # Initialize for this action
        for channel in range(channel_num):
            # Extract and standardize test data for the current channel
            channel_test_data = X_test[
                :, channel, :
            ]  # Adjust slicing based on X_test dimensions
            scaler = scalers_by_channel_action[action][channel]
            channel_test_data_scaled = scaler.transform(channel_test_data)

            # Get GMM for this action and channel
            gmm = gmms_by_channel_action[action][channel]

            # Compute probabilities for this channel
            channel_probabilities = gmm.score_samples(channel_test_data_scaled)

            # Accumulate probabilities (log probabilities can be added directly)
            action_probabilities += channel_probabilities

        # Store the total probability for this action
        probabilities[:, action - 1] = action_probabilities

    # Convert log probabilities to normal probabilities
    probabilities = np.exp(probabilities)

    # Normalize to get probabilities summing to 1 for each sample
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Compute the action with the highest probability for each test sample
    predicted_actions = (
        np.argmax(probabilities, axis=1) + 1
    )  # Add 1 to match action numbering

    # Now you can compare predicted_actions with y_test
    print("Predicted actions:", predicted_actions)
    print("Actual actions:", y_test)

    # Example of comparing predicted actions with the actual test labels
    accuracy = np.mean(predicted_actions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Example actions dictionary (replace with actual Action objects)
    actions = {
        "left_elbow_relax": Action(
            action_value=1, text="Left Elbow Relax", audio="", image=""
        ),
        "left_elbow_flex": Action(
            action_value=2, text="Left Elbow Flex", audio="", image=""
        ),
        "right_elbow_relax": Action(
            action_value=3, text="Right Elbow Relax", audio="", image=""
        ),
        "right_elbow_flex": Action(
            action_value=4, text="Right Elbow Flex", audio="", image=""
        ),
        "end_collection": Action(
            action_value=5, text="End Collection", audio="", image=""
        ),
    }

    # Example parameters (replace with actual values)
    subject_id = "108"
    visit_number = 1

    classify_eeg_data(subject_id, visit_number, actions)
