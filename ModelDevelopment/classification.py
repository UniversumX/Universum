import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from preprocessing import preprocess  # Import your preprocessing function
from scipy.stats import mode

from scipy.stats import mode




def extract_features(eeg_data, channels, freq_bands):
    """
    Extracts features from the preprocessed EEG data.
    We will take the mean power from each frequency band for specific channels.
    
    Args:
        eeg_data (numpy array): Preprocessed EEG data of shape (num_epochs, num_channels, num_frequency_bands, num_samples_per_epoch).
        channels (list): List of channel indices to extract features from.
        freq_bands (list): List of frequency band indices to extract features from.
    
    Returns:
        numpy array: Feature matrix of shape (num_epochs, num_features).
    """
    num_epochs, num_channels, num_frequency_bands, num_samples_per_epoch = eeg_data.shape

    # Ensure that the freq_bands indices are within bounds of the data
    valid_freq_bands = [band for band in freq_bands if band < num_frequency_bands]

    features = []
    for epoch in eeg_data:  # Iterate over epochs
        epoch_features = []
        for channel_idx in channels:  # Iterate over selected channels
            for band_idx in valid_freq_bands:  # Iterate over valid frequency bands
                # Compute the mean power across the time samples (amplitude values)
                power = np.mean(np.abs(epoch[channel_idx][band_idx]))  # Mean over time samples
                epoch_features.append(power)  # Append feature
        features.append(epoch_features)  # Append features for the epoch
    return np.array(features)


def load_data_and_labels(subject_id, visit_number, trial_number, actions):
    """
    Loads and preprocesses the EEG data from the specified directory.
    
    Args:
        subject_id (str): The subject ID.
        visit_number (int): The visit number.
        trial_number (int): The trial number.
        actions (dict): Dictionary mapping action names to their corresponding Action objects.
    
    Returns:
        tuple: (X, y) - Preprocessed feature matrix and corresponding labels.
    """
    # Construct the data directory path
    directory_path = f"../DataCollection/data/{subject_id}/{visit_number}/{trial_number}/"

    # Call the preprocess function from your preprocessing script
    eeg_data, accel_data, action_data = preprocess(directory_path, actions, should_visualize=False)

    # Define channel and frequency bands to use for feature extraction
    channels_to_use = [0, 1, 2, 3, 4, 5, 6, 7]  
    mu_band_idx = [8, 9, 10, 11]  # Example mu band indices (8-12 Hz)
    beta_band_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Beta band (13-30 Hz)
    freq_bands_to_use = mu_band_idx + beta_band_idx

    # Extract features from the EEG data
    X = extract_features(eeg_data, channels_to_use, freq_bands_to_use)

    # Get the labels from the action data
    y = action_data["action_value"].values  # Assuming action_data contains "action_value" column with labels 1, 2, 3, 4
    
    return X, y

def classify_eeg_data(subject_id, visit_number, trial_number, actions):
    # Load and preprocess data
    X, y = load_data_and_labels(subject_id, visit_number, trial_number, actions)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Debugging: Print the lengths of X and y to ensure they are consistent
    print(f"Length of X (features): {len(X_scaled)}")
    print(f"Length of y (labels): {len(y)}")

    # Ensure X and y have the same length
    min_length = min(len(X_scaled), len(y))
    X_scaled = X_scaled[:min_length]
    y = y[:min_length]

    # Now, split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


    # Train a Gaussian Mixture Model with 4 components (one for each class)
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm.fit(X_train)

    # Predict the class of the test data
    y_train_pred = gmm.predict(X_train)
    y_test_pred = gmm.predict(X_test)

    # Map GMM clusters to actual labels
    # This step is necessary because GMM assigns arbitrary labels to its components.
    # We use the training set to establish the mapping based on majority voting.

    # Assuming cluster is an array of indices for y_train

    mapping = {}
    for i in range(4):  # Assuming 4 clusters in GMM
        cluster = np.where(y_train_pred == i)[0]
        result = mode(y_train[cluster])

        if len(cluster) > 0:
            mapped_label = mode(y_train[cluster])[0]
            mapping[i] = mapped_label

    # Apply the mapping to predicted values
    y_train_mapped = np.array([mapping[cluster] for cluster in y_train_pred])
    y_test_mapped = np.array([mapping[cluster] for cluster in y_test_pred])

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_train_mapped)
    test_accuracy = accuracy_score(y_test, y_test_mapped)

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_test_mapped, target_names=["left elbow relax", "left elbow flex", "right elbow relax", "right elbow flex"]))

if __name__ == "__main__":
    # Example actions dictionary (replace with actual Action objects)
    from preprocessing import Action

    actions = {
        "left_elbow_relax": Action(action_value=1, text="Left Elbow Relax", audio="", image=""),
        "left_elbow_flex": Action(action_value=2, text="Left Elbow Flex", audio="", image=""),
        "right_elbow_relax": Action(action_value=3, text="Right Elbow Relax", audio="", image=""),
        "right_elbow_flex": Action(action_value=4, text="Right Elbow Flex", audio="", image=""),
    }

    # Example parameters (replace with actual values)
    subject_id = "105"
    visit_number = 1
    trial_number = 1

    classify_eeg_data(subject_id, visit_number, trial_number, actions)
