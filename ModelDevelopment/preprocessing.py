# Import necessary libraries for data processing, visualization, and handling EEG and other data types
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mne
import pandas as pd
from typing import Dict
import os
from loguru import logger

# Import specific functions from scipy for signal processing
# from DataCollection.actions import Action
from scipy import stats
from scipy import integrate
from scipy import signal
from sklearn.decomposition import PCA
import argparse

# Import dataclass to simplify class instantiation with named fields
from dataclasses import dataclass

# Define a dataclass to represent specific actions with associated data fields
@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str


# Set matplotlib's backend to "TkAgg" for compatibility, especially on certain OS configurations
# I (Matt) am running nixos and if I don't set this command then matplotlib won't show
matplotlib.use("TkAgg")

# Define a small constant to prevent division by zero in numerical computations
EPSILON = 1e-8


# Function to plot the Fast Fourier Transform (FFT) of the input data
def plot_fft(
    data, sampling_frequency, fft_window_size, percent_overlap_between_windows
):
    """
    This function just takes in the data and plots an fft of it
    Returns: The magnitude of the fft for each frequency
    """
    # Compute the spectrogram of the input data
    pxx, freqs, bins, im = plt.specgram(
        data,
        Fs=sampling_frequency,
        NFFT=fft_window_size,
        noverlap=int(percent_overlap_between_windows * fft_window_size),
    )
    # Display the plot
    plt.show()
    # Return the FFT results (power, frequencies, and time bins)
    return pxx, freqs, bins


# Function to plot entropy over time and frequency dimensions
def plot_entropy_of_data_time_and_frequncy_dimensions(pxx, freqs, times):
    """
    Plots the entropies of the data as a function of time, and a function of frequency.
    The purpose of this is the higher the entropy, the more random the data is so the more information the data has

    (Get these from plot_fft)
    pxx: The power spectral density of the data
    freqs: The frequencies that the pxx is at
    bins: The time bins that the pxx

    """
    # Calculate entropy across the frequency dimension
    frame_entropies = np.apply_along_axis(lambda x: stats.entropy(x + 1e-10), 0, pxx)
    # Calculate entropy across the time dimension
    freq_entropies = np.apply_along_axis(lambda x: stats.entropy(x + 1e-10), 1, pxx)


# Plot entropy of frequency bands
    plt.figure()
    plt.plot(freqs, freq_entropies)
    plt.title("Entropy of Frequency Bands")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Entropy")
    plt.show()


# Plot entropy of time frames
    plt.figure()
    plt.plot(times, frame_entropies)
    plt.title("Entropy of Time Frames")
    plt.xlabel("Time (s)")
    plt.ylabel("Entropy")

    plt.show()



# Function to time-align accelerometer data with EEG data using linear interpolation
def time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data):
    """
    Takes in the accelerometer data and time-aligns it with eeg data by lineraly interpoalting the accelerometer data
    """
    # Store the column names for later use
    accel_data_columns = accel_data.columns
    # Convert accelerometer data to numpy array for easier processing
    accel_data_np = accel_data.to_numpy()
    # Initialize an array to hold aligned data
    new_accel_data = np.zeros((len(eeg_data), accel_data_np.shape[1]))

    # Interpolate each accelerometer column to align with EEG timestamps
    for i in range(accel_data_np.shape[1]):
        new_accel_data[:, i] = np.interp(
            # Target timestamps (from EEG)
            eeg_data["timestamp"],
            # Original timestamps (from accelerometer)
            accel_data["timestamp"],
            # Data values to interpolate
            accel_data_np[:, i],
        )
    # Convert to DataFrame with original column names
    return pd.DataFrame(new_accel_data, columns=accel_data_columns)


# Function to load EEG, accelerometer, and action data from a specified visit of a subject
def get_data_from_visit(subject_id, trial_number, visit_number):
    # Load data as CSV

    # Define the directory path based on subject, visit, and trial identifiers
    data_directory_path = (
        f"../DataCollection/data/{subject_id}/{visit_number}/{trial_number}/"
    )

    # Load the data from respective CSV files
    eeg_data = pd.read_csv(data_directory_path + "eeg_data_raw.csv")
    accel_data = pd.read_csv(data_directory_path + "accelerometer_data.csv")
    action_data = pd.read_csv(data_directory_path + "action_data.csv")
    # Return the loaded data
    return eeg_data, accel_data, action_data


# Function to load data from a general directory path
def get_data_from_directory(data_directory_path: str):
    # Define file paths for EEG, accelerometer, and action data
    eeg_file = os.path.join(data_directory_path, "eeg_data_raw.csv")
    accel_file = os.path.join(data_directory_path, "accelerometer_data.csv")
    action_file = os.path.join(data_directory_path, "action_data.csv")
    
    # Check if all files exist; if not, log a warning and skip processing
    # Check if all required files exist
    if not os.path.exists(eeg_file) or not os.path.exists(accel_file) or not os.path.exists(action_file):
        logger.warning(f"One or more CSV files are missing in directory: {data_directory_path}. Skipping this folder.")
        # Return None to indicate missing files
        return None, None, None  # Return None to indicate skipping
    

    # Try loading the CSV files; if any error occurs, log it and return None
    # Try to read the CSV files
    try:
        eeg_data = pd.read_csv(eeg_file)
        accel_data = pd.read_csv(accel_file)
        action_data = pd.read_csv(action_file)
    except Exception as e:
        logger.warning(f"Error reading CSV files in {data_directory_path}: {e}")
        return None, None, None
    
    # Return loaded data
    return eeg_data, accel_data, action_data



# Function to convert timestamps in a DataFrame to seconds since the last epoch for consistency
def convert_timestamp_to_time_since_last_epoch(df):
    """
    Converts the timestamp to time since the last epoch
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) / 10**9
    return df


def align_data_to_experiment_start_and_end_time(df, start_time: float, end_time: float):
    """
    Aligns the data to the experiment start and end time
    """
    # Ensure that the end time is greater than the start time to avoid errors in filtering
    assert end_time > start_time
    # Filter the DataFrame rows based on the condition that timestamps fall within the start and end time.
    # The filtering condition creates a boolean mask:
    # - (df["timestamp"] >= start_time) is True for rows where the timestamp is >= start_time
    # - (df["timestamp"] <= end_time) is True for rows where the timestamp is <= end_time
    # The '&' operator combines these conditions, selecting rows where both are true.
    return df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]


def whiten_data_with_pca(data: np.array):
    """
    This function takes in a numpy array and then whitens it
    """
    # Create a copy of the data to avoid modifying the original array
    data = data.copy()

    # Subtract the mean of the data along each feature (column-wise) to center the data
    # This step is important in PCA as it removes the average signal, focusing on variance.
    data = data - data.mean()

    # Calculate the covariance matrix of the data to capture relationships between features
    # - data @ data.T performs a dot product, resulting in a (n_features, n_features) matrix.
    # - Dividing by data.shape[1] normalizes the covariance.
    noise_cov = data @ data.T / data.shape[1]

    # Perform eigen decomposition of the covariance matrix to get eigenvalues and eigenvectors
    # - Eigenvalues represent the amount of variance explained by each principal component.
    # - Eigenvectors represent the directions of the principal components.
    # find the eigenvectors of the covariance matrix to do PCA
    eigenvalues, eigenvectors = np.linalg.eig(noise_cov)

    # Create a diagonal matrix of the inverse square roots of the eigenvalues to normalize variances
    # - Adding EPSILON prevents division by zero in case of small or zero eigenvalues.
    inverse_lambda = np.diag(1 / (eigenvalues + EPSILON))
    return np.sqrt(inverse_lambda) @ eigenvectors.T @ data

def preprocess_person(directory_path: str, actions: Dict[str, Action], should_visualize=False):
    # Initialize an empty list to store the preprocessed data for each trial
    res = []
    # List all items in the specified directory. Each item represents a folder for a different trial
    items = os.listdir(directory_path)
    # Generate full paths to each item (trial folder) by combining the directory path with the item names
    full_paths = [os.path.join(directory_path, item) for item in items]

    # Loop through each full path, processing each trial folder separately
    for full_path in full_paths:
        # Call the 'preprocess' function on the current trial folder.
        # This function processes EEG, accelerometer, and action data within that folder.
        # The 'preprocess' function returns three objects: 
        # - x: preprocessed EEG data
        # - accel_data: accelerometer data
        # - action_data: action data
        x, accel_data, action_data = preprocess(full_path, actions, should_visualize)

        # Check if any of the returned data (EEG, accelerometer, or action data) is missing or None.
        # If any data is missing, skip this iteration and move on to the next trial folder.
        if x is None or accel_data is None or action_data is None:
            continue

        # Now safely check if x is empty
        if x.size == 0 or accel_data.empty or action_data.empty:
            continue  # Skip this iteration if any loaded data is empty

        # Append the valid preprocessed data for this trial to the 'res' list.
        # Each entry in 'res' is a list containing [EEG data, accelerometer data, action data] for one trial.
        res.append([x, accel_data, action_data])
    return res


#This function processes raw data from a single trial. 
#It aligns timestamps, applies filtering, and performs feature extraction (STFT and PCA).
def preprocess(directory_path: str, actions: Dict[str, Action], should_visualize=False):
    """
    Args:
    - directory_path: The path to the directory containing the data for a specific trial. Ex: "../DataCollection/data/103/1/1/"
    - actions: the actions that the user did
    - should_visualize: If we should visualize some plots (used for debugging)

    Returns:
    - eeg_data: Preprocessed EEG data, shape is (num_epochs, num_channels, frequency_bands, num_samples_per_epochs)
    - accel_data: Untouched accelerometer data
    - action_data: Untouched action data


    This function will run our preprocessing pipeline on a specific trial. This includes:
        - Aligning the timestamps of the data (so accel data and eeg data are aligned in time)
        - Interpolates accel data so that accel data datapoints align up with eeg data
        - Band filtering
        - STFT on eeg data

    """

    # Set parameters for EEG sampling and windowing
    # Make sure the sampling frequency is the sampling frequency said on the device
    # Hz, based on device documentation
    sampling_frequency = 256  # From the documentation
    # Window size for STFT
    window_size = 96
    # Percentage of overlap for windowed analysis
    percent_overlap = 0.95

    # Load EEG, accelerometer, and action data from the specified directory
    eeg_data, accel_data, action_data = get_data_from_directory(directory_path)
    if eeg_data is None or accel_data is None or action_data is None:
        # If any data is missing, return None for each to indicate failure
        return None, None, None

     # Now check if the loaded data is empty
    if eeg_data.empty or accel_data.empty or action_data.empty:
        return None, None, None  # Skip if any loaded data is empty
    
     # Fix the action data so if there are multiple actions of the same type after aech other, it removes those rows
    action_data = action_data.loc[
        (abs(action_data["action_value"] - action_data["action_value"].shift(1)) > 0)
    ].reset_index(drop=True)


    # Get rid of the end collection action data
    action_data = action_data[
        action_data["action_value"] != actions["end_collection"].action_value
    ]


    # Convert timestamps in each data type to "time since last epoch" format
    # Convert timestamp to time since last epoch (a float)
    eeg_data, accel_data, action_data = map(
        convert_timestamp_to_time_since_last_epoch, [eeg_data, accel_data, action_data]
    )

    # Drop the "device_id" column from accelerometer data since it's irrelevant here
    # Get rid of device id as we don't care about it
    accel_data = accel_data.drop(columns=["device_id"])

    # Define the experiment's start and end times, based on overlapping timestamps in EEG and accelerometer data
    # Make the timestamps so that they start and end at the same time, throw out data outside the starting/stopping times of each dataset
    experiment_start_time = max(
        [accel_data["timestamp"].iloc[0], eeg_data["timestamp"].iloc[0]]
    )
    experiment_end_time = min(
        [accel_data["timestamp"].iloc[-1], eeg_data["timestamp"].iloc[-1]]
    )

    # Trim data to only include records within the experiment's start and end times
    eeg_data, accel_data, action_data = map(
        lambda df: align_data_to_experiment_start_and_end_time(
            df, experiment_start_time, experiment_end_time
        ),
        [eeg_data, accel_data, action_data],
    )

    # Interpolate accelerometer data to align timestamps with EEG data
    accel_data = time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data)

    # Ensure the aligned EEG and accelerometer data have the same length
    assert len(eeg_data) == len(
        accel_data
    ), f"len(egg_data) != len(accel_data) ({len(eeg_data)} != {len(accel_data)})"
    assert len(action_data) > 0, "There is no action data!"

    # Prepare EEG channel names (column headers in EEG data) and types for MNE's RawArray
    # Time align the data by linearly interpolating the accelerometer data
    # Create column names (mne Raw Array needs this)
    # Exclude timestamp column
    ch_names = eeg_data.columns[1:].tolist()
    # All channels are EEG type
    ch_types = ["eeg"] * len(ch_names)


    # Convert action timestamps to sample indices and map actions to events
    events = []
    for index, row in action_data.iterrows():
        sample = np.argmin(np.abs(eeg_data["timestamp"] - row["timestamp"]))
        action_value = int(row["action_value"])
        events.append([sample, 0, action_value])

    # Convert to numpy array for MNE compatibility
    events = np.array(events)

    # Create a dictionary mapping action names to their respective action values
    event_dict = {
        action_name: action.action_value 
        for (action_name, action) in actions.items()
        if actions["end_collection"].action_value != action.action_value
    }

    # Set up EEG data with MNE using channel names, sampling frequency, and channel types
    info = mne.create_info(
        ch_names=ch_names, sfreq=sampling_frequency, ch_types=ch_types
    )
    # Transpose for MNE compatibility
    eeg_data_array = eeg_data[ch_names].to_numpy().T
    # Create a Raw object
    raw = mne.io.RawArray(eeg_data_array, info)


    # Apply a band-pass filter to remove noise outside the frequency range of interest
    # Apply a band filter to the data# Upper cutoff frequency in Hz
    # Upper cutoff frequency in Hz  
    cutoff_max = 45  # Cut off frequency for band filter
    # Lower cutoff frequency in Hz
    cutoff_min = 1  # Cut off frequency for band filter
    raw.filter(l_freq=cutoff_min, h_freq=cutoff_max, fir_design="firwin")

    # Segment (epoch) data into individual trials based on action events
    # Epoch the data so that every epoch is a trial
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        # Start time of each epoch
        # tmin=0,
        # Duration of each epoch in seconds
        tmax=2,
        # Baseline correction
        # baseline=(None, 0),
        preload=True,
    )

    # Extract epoch data as a numpy array and remove the last epoch (end of data collection)
    x = epochs.get_data(copy=True) # the last event is end of data collection
    # Extract event labels for each epoch
    y = epochs.events[:-1]  # the last event of the data

    # Reshape data to combine epochs for PCA (Principal Component Analysis)
    num_epochs, num_channels, num_samples = x.shape
    # Flatten epochs for PCA
    x = x.reshape(
        num_channels, num_epochs * num_samples
    )  # stack all of the epochs together for PCA

     # If visualization is enabled, plot events, epochs, and power spectral density
    if should_visualize:
        mne.viz.plot_events(events, sfreq=sampling_frequency, first_samp=0)
        plt.show()
        epochs.plot(events=events, event_id=event_dict)
        # Plot the epochs as an image map
        # Plot EEG epochs as images
        epochs.plot_image(picks="eeg")
        fig = raw.compute_psd(tmax=np.inf, fmax=sampling_frequency // 2).plot(
            average=False, amplitude=False, picks="data", exclude="bads"
        )
        plt.show()

    # Whiten data using PCA to remove correlations between channels
    # whiten the data with PCA
    whitened_data = whiten_data_with_pca(x)


    # STFT should be done per channel, per epoch, not mixed.
    x = whitened_data.reshape(num_channels, num_epochs, num_samples).transpose(1, 0, 2)

    # Run stft on the data
    x = np.apply_along_axis(
        lambda x: signal.stft(
            x,
            fs=sampling_frequency,
            nperseg=window_size,
            noverlap=window_size * percent_overlap,
        )[2],
        2,
        x,
    )


    # get rid of frequencies above cutoff_max frequency
    top_index = int(np.ceil(x.shape[2] / ((sampling_frequency / 2) / cutoff_max)))
    x = x[:, :, :top_index, :]

    # sweeping window to increase data
    num_epochs, num_channels, num_frequencies, num_samples = x.shape


    # Do PCA on the data in a feature extraction portion
    # num_components = 32
    # features = np.zeros((num_channels, num_components, num_samples))
    #
    # # do a spectogram of the data
    # # okay, so basically we gotta decide how to do PCA on this dataset, if we make the dimension of the PCA be frequencies * channels then running PCA
    # # then PCA will find components for each individual channel, but if we instead have the dimension just be frequencies, then PCA will be finding components
    # # for the channels at the same time, so the dimension of the eigenvectors will be lower, and we will get less characteristics of each
    # # channel. tbh idk what is the best to do. intuitively, it would be better for PCA dimension to be frequencies * channels if we had more data.
    # print(x.shape)
    # for channel in range(x.shape[0]):
    #     # so this plots the spectogram, it should be:
    #     plt.figure()
    #     plt.imshow(10 * np.log10(np.abs(x)).T, aspect="auto", origin="lower")
    #     plt.title(f"Spectrogram of Channel {channel}")
    #     plt.ylabel("Frequency * Epoch [Hz]")
    #     plt.xlabel("Time [s]")
    #     plt.show()
    #
    # features = PCA(n_components=2).fit_transform(stft_data.T)

    # plt.scatter(features[:, 0], features[:, 1])
    # plt.show()

    # ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=800)
    # ica.fit(whitened_raw)
    # ica.plot_sources(whitened_raw, show_scrollbars=False)
    # plt.show()
    #
    # def print_relative_importance_of_ICA_features(ica):
    #     for i, component in enumerate(ica.mixing_matrix_):
    #         explained_var_ratio = ica.get_explained_variance_ratio(
    #             whitened_raw, components=[i], ch_type="eeg"
    #         )
    #         print(
    #             f"Fraction of variance in EEG signal explained by {i}th component: {explained_var_ratio['eeg']}"
    #         )
    #
    # print_relative_importance_of_ICA_features(ica)
    #
    # sources = ica.get_sources(whitened_raw).get_data()
    # first_component_signal = sources[0, :]

    # print(x.shape)
    # print(accel_data.shape)
    # print(action_data.shape)


    # Return preprocessed EEG data, along with untouched accelerometer and action data
    return x, accel_data, action_data["action_value"]


if __name__ == "__main__":
    # Checks if this script is being run directly
    ## Replace this with reading from the study
    # Defines a dictionary called 'actions' to store different action types and their metadata.
    actions = {
        # Creates an action for flexing the left elbow
        "left_elbow_flex": Action(
            # Unique integer identifier for this action
            action_value=1,
            text="Please flex your left elbow so your arm raises to shoulder level",
            # Placeholder path for an audio file with instructions
            audio="path/to/audio",
            # Placeholder path for an image illustrating the action
            image="path/to/image",
        ),

        # Creates an action for relaxing the left elbow
        "left_elbow_relax": Action(
            # Unique integer identifier for this action
            action_value=2,
            # Instruction text for the action
            text="Please relax your left elbow back to original state",
            # Placeholder path for an audio file with instructions
            audio="path/to/audio",
            # Placeholder path for an image illustrating the action
            image="path/to/image",
        ),

        # Creates an action for flexing the right elbow
        "right_elbow_flex": Action(
            # Unique integer identifier for this action
            action_value=3,
            text="Please flex your right elbow so your arm raises to shoulder level",
            # Placeholder path for an audio file with instructions
            audio="path/to/audio",
            # Placeholder path for an image illustrating the action
            image="path/to/image",
        ),

        # Creates an action for relaxing the right elbow
        "right_elbow_relax": Action(
            action_value=4,
            text="Please relax your right elbow back to original state",
            audio="path/to/audio",
            image="path/to/image",
        ),

        # Creates an action marking the end of data collection
        "end_collection": Action(
            # Unique integer identifier for this action
            action_value=5, 
            text="Data collection ended", 
            # No audio file required for this action
            audio=None, 
            # No image file required for this action
            image=None
        ),
    }
    
    # Specifies the identifiers for the subject and visit number of the dataset
    # Unique identifier for the subject in the study
    subject_id = 105
    # Identifier for a specific visit or session with the subject
    visit_number = 1

    # Calls the 'preprocess_person' function to preprocess data for the specified subject and visit
    # - 'f"../DataCollection/data/{subject_id}/{visit_number}/"': Path to the directory containing the subject's data
    # - 'actions': Dictionary containing all actions for this study
    # - 'should_visualize=False': Disables data visualization during processing
    res = preprocess_person(
        f"../DataCollection/data/{subject_id}/{visit_number}/",
        actions,
        should_visualize=False,
    )
    
    # Iterates through the preprocessed data returned by 'preprocess_person' function
    for eeg_data, accel_data, action_data in res:
        print("Eeg data shape:", eeg_data.shape)
        print(action_data)





# model = NMF(n_components=n_components, init="random", random_state=0)
# W = model.fit_transform(stft_data.T)
# H = model.components_
#
# # Plot NMF components
# plt.figure(figsize=(15, 10))
# for i in range(n_components):
#     plt.subplot(n_components // 2, 2, i + 1)
#     plt.plot(H[i])
#     plt.title(f"NMF Component {i+1}")
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
# plt.tight_layout()
# plt.show()
#
# # Print relative importance of NMF features
# print("Relative importance of NMF features:")
# feature_importance = np.sum(W, axis=0)
# feature_importance /= np.sum(feature_importance)
# for i, importance in enumerate(feature_importance):
#     print(f"Feature {i+1}: {importance:.4f}")
#
# # Reconstruct the signal using NMF components
# reconstructed_data = np.dot(W, H)
#
# # Calculate reconstruction error
# reconstruction_error = np.mean((stft_data.T - reconstructed_data) ** 2)
# print(f"Reconstruction error: {reconstruction_error:.4f}")
#
# # Plot original vs reconstructed signal for the first channel
# plt.figure(figsize=(15, 5))
# plt.plot(stft_data[0], label="Original")
# plt.plot(reconstructed_data[:, 0], label="Reconstructed")
# plt.title("Original vs Reconstructed Signal (First Channel)")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.show()
#
# # Perform FFT on NMF components
# for i in range(n_components):
#     plot_fft(H[i], sfreq, NFFT, percent_overlap)
#     plt.title(f"FFT of NMF Component {i+1}")
#     plt.show()


### Explort preprocessed data

# do a tsne on the data
