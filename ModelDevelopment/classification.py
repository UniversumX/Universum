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
    eeg_data, accel_data, action_data = preprocess_person(
        directory_path,
        actions,
        should_visualize=False,
    )

    # Print the number of epochs in eeg_data
    num_epochs = eeg_data.shape[0]
    print(f"Number of epochs in EEG data: {num_epochs}")

    # Print the number of entries in action_data
    num_action_entries = len(action_data)
    print(f"Number of action data entries: {num_action_entries}")

    # Extract the number of samples per epoch (from the last dimension of eeg_data)
    num_samples_per_epoch = eeg_data.shape[-1]

    Fs = 256  # Sampling frequency in Hz

    # Generate frequency values for positive frequencies only (assuming real-valued EEG data)
    frequencies = np.fft.rfftfreq(num_samples_per_epoch, d=1 / Fs)

    # Define channels
    channels_to_use = [0, 1, 2, 3, 4, 5, 6, 7]

    # Extract features
    X = extract_features(eeg_data, channels_to_use, frequencies)
    y = action_data  # Assuming action_data contains "action_value" column with labels 1, 2, 3, 4

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

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Now, split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train a Gaussian Mixture Model with 4 components (one for each class)
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm.fit(X_train)
    # gmm.fit(X_scaled)
    # Predict the class of the test data
    # test = gmm.predict(X_scaled)
    y_train_pred = gmm.predict(X_train)
    y_test_pred = gmm.predict(X_test)
    """
    # Reduce X_scaled to 2 components using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Scatter plot of the reduced features
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=test, cmap='viridis', edgecolor='k', s=50)
    plt.colorbar(label='Class Label')
    plt.show()
    """

    # Map GMM clusters to actual labels
    mapping = {}
    for i in range(4):  # Assuming 4 clusters in GMM
        cluster = np.where(y_train_pred == i)[0]
        result = mode(y_train[cluster])

    # Apply the mapping to predicted values
    y_train_mapped = np.array([mapping[cluster] for cluster in y_train_pred])
    y_test_mapped = np.array([mapping[cluster] for cluster in y_test_pred])

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_train_mapped)
    test_accuracy = accuracy_score(y_test, y_test_mapped)

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_test_mapped,
            target_names=[
                "left elbow relax",
                "left elbow flex",
                "right elbow relax",
                "right elbow flex",
            ],
        )
    )


if __name__ == "__main__":
    # Example actions dictionary (replace with actual Action objects)
    from preprocessing import Action

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
    subject_id = "105"
    visit_number = 1

    classify_eeg_data(subject_id, visit_number, actions)
