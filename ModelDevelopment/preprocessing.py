import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mne
import pandas as pd
from typing import Dict
import os
from loguru import logger

# from DataCollection.actions import Action
from scipy import stats
from scipy import integrate
from scipy import signal
from sklearn.decomposition import PCA
import argparse


from dataclasses import dataclass


@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str


# I (Matt) am running nixos and if I don't set this command then matplotlib won't show
matplotlib.use("TkAgg")

EPSILON = 1e-8


def plot_fft(
    data, sampling_frequency, fft_window_size, percent_overlap_between_windows
):
    """
    This function just takes in the data and plots an fft of it
    Returns: The magnitude of the fft for each frequency
    """
    pxx, freqs, bins, im = plt.specgram(
        data,
        Fs=sampling_frequency,
        NFFT=fft_window_size,
        noverlap=int(percent_overlap_between_windows * fft_window_size),
    )
    plt.show()
    return pxx, freqs, bins


def plot_entropy_of_data_time_and_frequncy_dimensions(pxx, freqs, times):
    """
    Plots the entropies of the data as a function of time, and a function of frequency.
    The purpose of this is the higher the entropy, the more random the data is so the more information the data has

    (Get these from plot_fft)
    pxx: The power spectral density of the data
    freqs: The frequencies that the pxx is at
    bins: The time bins that the pxx

    """
    frame_entropies = np.apply_along_axis(lambda x: stats.entropy(x + 1e-10), 0, pxx)
    freq_entropies = np.apply_along_axis(lambda x: stats.entropy(x + 1e-10), 1, pxx)

    plt.figure()
    plt.plot(freqs, freq_entropies)
    plt.title("Entropy of Frequency Bands")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Entropy")
    plt.show()

    plt.figure()
    plt.plot(times, frame_entropies)
    plt.title("Entropy of Time Frames")
    plt.xlabel("Time (s)")
    plt.ylabel("Entropy")

    plt.show()


def time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data):
    """
    Takes in the accelerometer data and time-aligns it with eeg data by lineraly interpoalting the accelerometer data
    """
    accel_data_columns = accel_data.columns
    accel_data_np = accel_data.to_numpy()
    new_accel_data = np.zeros((len(eeg_data), accel_data_np.shape[1]))
    for i in range(accel_data_np.shape[1]):
        new_accel_data[:, i] = np.interp(
            eeg_data["timestamp"],
            accel_data["timestamp"],
            accel_data_np[:, i],
        )
    return pd.DataFrame(new_accel_data, columns=accel_data_columns)


def get_data_from_visit(subject_id, trial_number, visit_number):
    # Load data as CSV

    data_directory_path = (
        f"../DataCollection/data{subject_id}/{visit_number}/{trial_number}/"
    )

    eeg_data = pd.read_csv(data_directory_path + "eeg_data_raw.csv")
    accel_data = pd.read_csv(data_directory_path + "accelerometer_data.csv")
    action_data = pd.read_csv(data_directory_path + "action_data.csv")
    return eeg_data, accel_data, action_data


def get_data_from_directory(data_directory_path: str):
    eeg_file = os.path.join(data_directory_path, "eeg_data_raw.csv")
    accel_file = os.path.join(data_directory_path, "accelerometer_data.csv")
    action_file = os.path.join(data_directory_path, "action_data.csv")

    # Check if all required files exist
    if (
        not os.path.exists(eeg_file)
        or not os.path.exists(accel_file)
        or not os.path.exists(action_file)
    ):
        logger.warning(
            f"One or more CSV files are missing in directory: {data_directory_path}. Skipping this folder."
        )
        return None, None, None  # Return None to indicate skipping

    # Try to read the CSV files
    try:
        eeg_data = pd.read_csv(eeg_file)
        accel_data = pd.read_csv(accel_file)
        action_data = pd.read_csv(action_file)
    except Exception as e:
        logger.warning(f"Error reading CSV files in {data_directory_path}: {e}")
        return None, None, None

    return eeg_data, accel_data, action_data


def convert_timestamp_to_time_since_last_epoch(df):
    """
    Converts the timestamp to time since the last epoch
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype('int64') / 10**9
    return df


def align_data_to_experiment_start_and_end_time(df, start_time: float, end_time: float):
    """
    Aligns the data to the experiment start and end time
    """
    assert end_time > start_time
    return df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]


def whiten_data_with_pca(data: np.array):
    """
    This function takes in a numpy array and then whitens it
    """
    data = data.copy()
    data = data - data.mean()

    noise_cov = data @ data.T / data.shape[1]

    # find the eigenvectors of the covariance matrix to do PCA
    eigenvalues, eigenvectors = np.linalg.eig(noise_cov)
    inverse_lambda = np.diag(1 / (eigenvalues + EPSILON))
    return np.sqrt(inverse_lambda) @ eigenvectors.T @ data


def preprocess_person(
    directory_path: str, actions: Dict[str, Action], should_visualize=False
):
    res = []
    items = os.listdir(directory_path)
    full_paths = [os.path.join(directory_path, item) for item in items]
    for full_path in full_paths:
        x, accel_data, action_data = preprocess(full_path, actions, should_visualize)
        if x is None or accel_data is None or action_data is None:
            continue
        res.append([x, accel_data, action_data])
    return res

def feature_extract(x):
    num_epochs, num_channels, num_frequencies, num_samples = x.shape

    # set window size and offset
    window_size = 10
    window_offset = 1
    y = []

    # sweeping window
    i = 0
    while i <= num_samples - window_size:
        y.append(x[:, :, :, i: i+window_size])
        i += window_offset
    y = np.stack(y, axis=4)
    
    print("windowed dimensions:", y.shape) # num_epochs, num_channels, num_frequencies, window_size, num_window
    print("original dimensions: ", x.shape)

    a = num_frequencies * window_size # reused variable
    eigvecs = []
    eigvals = []
    for j in range(y.shape[0]):
        vecs_epoch = []
        vals_epoch = []
        for k in range(y.shape[1]):
            y_1 = y[j][k]
            y_flattened = y_1.reshape(-1, y_1.shape[2]).transpose()

            # PCA
            y_standardized = (y_flattened - np.mean(y_flattened, axis=0)) / y_flattened.std(axis=0)
            cov = np.cov(y_standardized, rowvar=False)
            eigval, eigvec = np.linalg.eigh(cov)
            sorted_indices= np.argsort(eigval)[::-1]
            eigval = eigval[sorted_indices]
            eigvec = eigvec[:, sorted_indices]

            vecs_epoch.append(eigvec)
            vals_epoch.append(eigval)
        #print(vals_epoch.shape)
        #print(eigvals.shape)
        eigvecs.append(vecs_epoch)
        eigvals.append(vals_epoch)
    
    eigvecs = np.array(eigvecs)
    eigvals = np.array(eigvals)

    #TODO: project data onto eigenspace
    
    # print("eigenvectors:", eigvecs.shape)
    # print("eigenvalues:", eigvals.shape)

def snr(signal):
    signal_power = np.mean(signal**2)
    
    #assume noise is deviation from mean
    noise = signal - np.mean(signal)
    noise_power = np.mean(noise**2)
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_snr(eeg_data):
    num_epochs, num_channels, num_samples = eeg_data.shape
    snr_values = np.zeros((num_epochs, num_channels))
    
    # Loop through each epoch and each channel
    for epoch in range(num_epochs):
        for channel in range(num_channels):
            snr_values[epoch, channel] = snr(eeg_data[epoch, channel])
    
    return snr_values


def combine_snrs(snr_values):
    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_values / 10)
    # Compute the average of the linear SNRs
    avg_snr_linear = np.mean(snr_linear)
    # Convert the average linear SNR back to dB
    combined_snr_db = 10 * np.log10(avg_snr_linear)
    
    return combined_snr_db


def wiener_filter(x, type, mysize=None, noise=None):
    #1. direct
    #2. by channel
    #3. by epoch
    #4. by epoch and channel
    num_epochs, num_channels, num_samples = x.shape
    if (type == 1):
        # Combine everything
        combined = x.reshape(-1)
        filtered_combined = signal.wiener(combined, mysize=mysize, noise=noise)
        filtered = filtered_combined.reshape(x.shape)
        return filtered
    elif (type == 2):
        filtered = np.zeros_like(x)
        for channel in range(num_channels):
            # Combine all epochs for this channel into one continuous signal
            combined = x[:, channel, :].reshape(-1)
            filtered_channel = signal.wiener(combined, mysize=mysize, noise=noise)
            filtered[:, channel, :] = filtered_channel.reshape(num_epochs, num_samples)
        return filtered
    elif (type == 3):
        filtered = np.zeros_like(x)
        for epoch in range(num_epochs):
            combined = x[epoch, :, :].reshape(-1)
            # Combine all channels for this epoch into one continuous signal
            filtered_epoch = signal.wiener(combined, mysize=mysize, noise=noise)
            filtered[epoch, :, :] = filtered_epoch.reshape(num_channels, num_samples)
        return filtered
    else: #type 4
        filtered = np.copy(x)
        for epoch in range(num_epochs):
            for channel in range(num_channels):
                # Apply filter to each channel of each epoch
                filtered[epoch, channel] = signal.wiener(x[epoch, channel], mysize=mysize, noise=noise)
        return filtered

def preprocess(directory_path: str, actions: Dict[str, Action], should_visualize=False):
    """
    Args:
    - directory_path: The path to the directory containing the data for a specific trial. Ex: "../DataCollection/data/103/"
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

    # Make sure the sampling frequency is the sampling frequency said on the device
    sampling_frequency = 256  # From the documentation
    window_size = 96
    percent_overlap = 0.95

    eeg_data, accel_data, action_data = get_data_from_directory(directory_path)
    if eeg_data is None or accel_data is None or action_data is None:
        return None, None, None

    # Fix the action data so if there are multiple actions of the same type after aech other, it removes those rows
    action_data = action_data.loc[
        (abs(action_data["action_value"] - action_data["action_value"].shift(1)) > 0)
    ].reset_index(drop=True)

    # Get rid of the end collection action data
    action_data = action_data[
        action_data["action_value"] != actions["end_collection"].action_value
    ]

    # Convert timestamp to time since last epoch (a float)
    eeg_data, accel_data, action_data = map(
        convert_timestamp_to_time_since_last_epoch, [eeg_data, accel_data, action_data]
    )

    # Get rid of device id as we don't care about it
    accel_data = accel_data.drop(columns=["device_id"])

    # Make the timestamps so that they start and end at the same time, throw out data outside the starting/stopping times of each dataset
    experiment_start_time = max(
        [accel_data["timestamp"].iloc[0], eeg_data["timestamp"].iloc[0]]
    )
    experiment_end_time = min(
        [accel_data["timestamp"].iloc[-1], eeg_data["timestamp"].iloc[-1]]
    )

    eeg_data, accel_data, action_data = map(
        lambda df: align_data_to_experiment_start_and_end_time(
            df, experiment_start_time, experiment_end_time
        ),
        [eeg_data, accel_data, action_data],
    )

    accel_data = time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data)

    assert len(eeg_data) == len(
        accel_data
    ), f"len(egg_data) != len(accel_data) ({len(eeg_data)} != {len(accel_data)})"
    assert len(action_data) > 0, "There is no action data!"

    # Time align the data by linearly interpolating the accelerometer data
    # Create column names (mne Raw Array needs this)
    ch_names = eeg_data.columns[1:].tolist()
    ch_types = ["eeg"] * len(ch_names)


    events = []
    eeg_data["timestamp"] = pd.to_datetime(eeg_data["timestamp"])
    action_data["timestamp"] = pd.to_datetime(action_data["timestamp"])
    last_eeg_timestamp = eeg_data["timestamp"].max()
    threshold = pd.Timedelta(seconds=0.1)

    for index, row in action_data.iterrows():
        sample = np.argmin(np.abs(eeg_data["timestamp"] - row["timestamp"]))
        action_value = int(row["action_value"])
        # Check if the action_data timestamp is too far past the last eeg_data timestamp (0.1 seconds)
        if row["timestamp"] > last_eeg_timestamp + threshold:
            print(f"Warning: Action data timestamp {row['timestamp']} is more than {threshold} past the last EEG timestamp {last_eeg_timestamp}")
        events.append([sample, 0, action_value])

    events = np.array(events)
    event_dict = {
        action_name: action.action_value
        for (action_name, action) in actions.items()
        if actions["end_collection"].action_value != action.action_value
    }

    info = mne.create_info(ch_names=ch_names, sfreq=sampling_frequency, ch_types=ch_types)
    eeg_data_array = eeg_data[ch_names].to_numpy().T
    raw = mne.io.RawArray(eeg_data_array, info)

    # Apply a band filter to the data
    cutoff_max = 45  # Cut off frequency for band filter
    cutoff_min = 1  # Cut off frequency for band filter
    raw.filter(l_freq=cutoff_min, h_freq=cutoff_max, fir_design="firwin")

    # Epoch the data so that every epoch is a trial
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        # tmin=0,
        tmax=2,
        # baseline=(None, 0),
        preload=True,
    )
    # TODO: Move this to extract features function
    x = epochs.get_data(copy=True)  # the last event is end of data collection
    y = epochs.events[:-1]  # the last event of the data
    num_epochs, num_channels, num_samples = x.shape

    #filter
    snr_before_filtering = compute_snr(x)
    x = wiener_filter(x, 1) # toggle second arg [1-4] (works marginally better with 1-3)
    snr_after_filtering = compute_snr(x) # evaluating filter performance with signal to noise ratio
    print(f"Combined SNR Before Filtering: {combine_snrs(snr_before_filtering):.2f} dB")
    print(f"Combined SNR After Filtering: {combine_snrs(snr_after_filtering):.2f} dB")

    x = x.reshape(
        num_channels, num_epochs * num_samples
    )  # stack all of the epochs together for PCA

    if should_visualize:
        mne.viz.plot_events(events, sfreq=sampling_frequency, first_samp=0)
        plt.show()
        epochs.plot(events=events, event_id=event_dict)
        # Plot the epochs as an image map
        epochs.plot_image(picks="eeg")
        fig = raw.compute_psd(tmax=np.inf, fmax=sampling_frequency // 2).plot(
            average=False, amplitude=False, picks="data", exclude="bads"
        )
        plt.show()

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


    feature_extract(x)

    # stack the epochs together for PCA
    if should_visualize:
        # Compute pxx, frequencies, and times using plot_fft
        pxx, frequencies, times = plot_fft(
        data=eeg_data[ch_names].to_numpy().flatten(),
        sampling_frequency=sampling_frequency,
        fft_window_size=window_size,
        percent_overlap_between_windows=percent_overlap,
    )
        plot_entropy_of_data_time_and_frequncy_dimensions(pxx, frequencies, times)

    # Do PCA on the data in a feature extraction portion
    num_components = 32
    atures = np.zeros((num_channels, num_components, num_samples))
    print(x.shape)
    return x, accel_data, action_data["action_value"]


if __name__ == "__main__":
    ## Replace this with reading from the study
    actions = {
        "left_elbow_flex": Action(
            action_value=1,
            text="Please flex your left elbow so your arm raises to shoulder level",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "left_elbow_relax": Action(
            action_value=2,
            text="Please relax your left elbow back to original state",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "right_elbow_flex": Action(
            action_value=3,
            text="Please flex your right elbow so your arm raises to shoulder level",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "right_elbow_relax": Action(
            action_value=4,
            text="Please relax your right elbow back to original state",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "end_collection": Action(
            action_value=5, text="Data collection ended", audio=None, image=None
        ),
    }

    
    subject_id = 105
    visit_number = 1
    res = preprocess_person(
        f"../DataCollection/data/{subject_id}/{visit_number}/",
        actions,
        should_visualize=False,
    )
    for eeg_data, accel_data, action_data in res:
        print("Eeg data shape:", eeg_data.shape)
        print(action_data)


