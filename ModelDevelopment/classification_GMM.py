import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from preprocessing import preprocess_person  # Import your preprocessing function
from scipy.stats import mode
from sklearn.decomposition import PCA
from typing import Dict
from utils import *
from dataclasses import dataclass


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
    # eeg_data, accel_data, action_data 
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

    # Extract features
    # X = extract_features(eeg_feature, channels_to_use, frequencies)
    X = np.array(eeg_feature_combined)
    y = np.array(action_data_combined)

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
    X, y = load_data_and_labels(subject_id, visit_number, actions) # X = [num_epoch,num_channel,eigvec, num_component]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # # Instantiate model and train
    # model = SVC(kernel="rbf", C=1.0, gamma="scale")
    # model.fit(X_train, y_train)

    # # Determine accuraacy of model
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")

    # # Save model to specific path
    # model_dir = f"models/{subject_id}/{visit_number}"
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, "svm_model.joblib")
    # joblib.dump(model, model_path)
    # print(f"Model saved to: {model_path}")

    action_num = 4
    channel_num = 8

    # Initialize a structure to hold arrays for each channel, action, and epoch
    features_by_channel_action = {
        action: {channel: [] for channel in range(channel_num)} for action in range(1, action_num + 1)
    }

    # Iterate over actions
    for action in range(1, action_num + 1):
        # Filter data for the current action
        action_data = X_train[y_train == action]

        # Iterate over channels
        for channel in range(channel_num):
            # Iterate over epochs
            for epoch in range(action_data.shape[0]):
                # Extract features for the current epoch and channel
                epoch_features = action_data[epoch, channel, :, :]  # Shape: [eigvec, num_component]

                # Append the features to the list
                features_by_channel_action[action][channel].append(epoch_features)


    # Initialize a dictionary to store GMMs for each action and channel
    gmms_by_channel_action = {
        action: {channel: None for channel in range(channel_num)} for action in range(1, action_num + 1)
    }

    # Initialize a dictionary to store scalers for each action and channel
    scalers_by_channel_action = {
        action: {channel: [] for channel in range(channel_num)} for action in range(1, action_num + 1)
    }
    # Train a GMM for each action and channel
    for action in range(1, action_num + 1):
        for channel in range(channel_num):
            # Extract data for the current action and channel
            channel_data = features_by_channel_action[action][channel]  # List of features for all epochs

            # Ensure channel_data is not empty
            if not channel_data:
                raise ValueError(f"No data found for action {action}, channel {channel}")

            # Standardize data epoch by epoch
            channel_scaled_data = []
            scalers = []

            for epoch_idx, epoch_features in enumerate(channel_data):  # Loop over each epoch
                if epoch_features is None or epoch_features.size == 0:
                    print(f"Skipping empty epoch for action {action}, channel {channel}, epoch {epoch_idx}")
                    continue

                scaler = StandardScaler()
                epoch_features_scaled = scaler.fit_transform(np.abs(epoch_features))  # Standardize the current epoch

                if epoch_features_scaled is None or epoch_features_scaled.size == 0:
                    raise ValueError(f"Invalid scaled data for action {action}, channel {channel}, epoch {epoch_idx}")

                channel_scaled_data.append(epoch_features_scaled)  # Append scaled data
                scalers.append(scaler)  # Store the scaler for this epoch

            # Combine all scaled data for this action and channel
            if not channel_scaled_data:
                raise ValueError(f"No valid scaled data for action {action}, channel {channel}")

            # Convert to NumPy array and ensure it's 2D
            channel_scaled_data = np.vstack(channel_scaled_data)  # Shape: [num_epochs * num_features, num_features]

            # Validate the shape of the data
            if len(channel_scaled_data.shape) != 2:
                raise ValueError(f"Invalid shape for channel_scaled_data: {channel_scaled_data.shape}")
            if channel_scaled_data.shape[0] < 1 or channel_scaled_data.shape[1] < 1:
                raise ValueError(f"Insufficient samples or features in channel_scaled_data: {channel_scaled_data.shape}")

            # Train a GMM with 1 component for this action and channel
            gmm = GaussianMixture(n_components=1, random_state=42)
            gmm.fit(channel_scaled_data)  # Ensure this data is 2D and non-empty


            # Store the trained GMM and scalers
            gmms_by_channel_action[action][channel] = gmm
            scalers_by_channel_action[action][channel] = scalers

    print(f"Channel scaled data shape: {channel_scaled_data.shape}")

    predicted_actions = []

    for epoch in range(X_test.shape[0]):
        # Extract test data for the current epoch
        epoch_test_data = X_test[epoch]  # Shape: [num_channels, eigvec, num_component]
        epoch_test_scaled_data = []

        for channel in range(channel_num):
            # Retrieve the scaler for this channel and action (from training)
            channel_scaled_data = []
            for action in range(1, action_num + 1):
                scaler = scalers_by_channel_action[action][channel][epoch % len(scalers_by_channel_action[action][channel])]
                
                # Standardize the test data for the current channel
                channel_data = scaler.transform(np.abs(epoch_test_data[channel]))
                channel_scaled_data.append(channel_data)

            # Combine the scaled data for all channels
            epoch_test_scaled_data.append(np.vstack(channel_scaled_data))

        epoch_test_scaled_data = np.array(epoch_test_scaled_data)  # Ensure correct shape
        # print(f"Test scaled data shape: {epoch_test_scaled_data.shape}")

        # Initialize probabilities for this epoch
        probabilities = np.zeros((action_num, channel_num))  # [action_num, channel_num]

        for channel in range(channel_num):
            # Calculate probabilities for each action for this channel
            for action in range(1, action_num + 1):
                # Get the GMM for this action and channel
                gmm = gmms_by_channel_action[action][channel]

                # Compute the log probability for the scaled data
                log_probability = gmm.score_samples(epoch_test_scaled_data[channel])
                probabilities[action - 1, channel] += np.sum(log_probability)

        # Aggregate probabilities across all channels
        combined_action_probabilities = probabilities.sum(axis=1)  # Sum over channels for each action

        # Find the most likely action for this epoch
        most_likely_action = np.argmax(combined_action_probabilities) + 1  # Add 1 to match action numbering
        predicted_actions.append(most_likely_action) 

    predicted_actions = np.array(predicted_actions)  # Convert list to NumPy array
    accuracy = np.mean(predicted_actions == y_test)  # Compute accuracy

    # Print predicted and actual actions side by side
    print("Epoch | Predicted Action | Actual Action")
    print("----------------------------------------")
    for epoch, (predicted, actual) in enumerate(zip(predicted_actions, y_test)):
        print(f"{epoch + 1:5d} | {predicted:16d} | {actual:12d}")

    # Print the final accuracy
    print(f"\nAccuracy: {accuracy * 100:.2f}%")



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
    subject_id = "110"
    visit_number = 1

    classify_eeg_data(subject_id, visit_number, actions)
