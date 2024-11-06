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

    # data_directory_path = (
    #     f"../DataCollection/data/{subject_id}/{visit_number}/{trial_number}/"
    # )
    data_directory_path = (
        f"../DataCollection/data/103/4/"
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
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) / 10**9
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

### TODO:
# Read the action_data.csv and use mne events to label specific events in the data, incorporate that with the `raw` variable
# Create events from action_data
# events = []
# for index, row in action_data.iterrows():
#     sample = np.argmin(np.abs(eeg_data["timestamp"] - row["timestamp"]))
#     action_value = int(row["action_value"])
#     events.append([sample, 0, action_value])
# #TODO: put an assert message in here that if the action_data timestamp is too far past the last eeg_data timestamp then it console prints a message


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

    info = mne.create_info(
        ch_names=ch_names, sfreq=sampling_frequency, ch_types=ch_types
    )
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
    x = epochs.get_data(copy=True)  # the last event is end of data collection
    y = epochs.events[:-1]  # the last event of the data
    num_epochs, num_channels, num_samples = x.shape
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
    # <<<<<<< HEAD
    #
    #     # # do a spectogram of the data
    #     # # okay, so basically we gotta decide how to do PCA on this dataset, if we make the dimension of the PCA be frequencies * channels then running PCA
    #     # # then PCA will find components for each individual channel, but if we instead have the dimension just be frequencies, then PCA will be finding components
    #     # # for the channels at the same time, so the dimension of the eigenvectors will be lower, and we will get less characteristics of each
    #     # # channel. tbh idk what is the best to do. intuitively, it would be better for PCA dimension to be frequencies * channels if we had more data.
    #     # print(x.shape)
    #     # for channel in range(x.shape[0]):
    #     #     # so this plots the spectogram, it should be:
    #     #     plt.figure()
    #     #     plt.imshow(10 * np.log10(np.abs(x)).T, aspect="auto", origin="lower")
    #     #     plt.title(f"Spectrogram of Channel {channel}")
    #     #     plt.ylabel("Frequency * Epoch [Hz]")
    #     #     plt.xlabel("Time [s]")
    #     #     plt.show()
    #     #
    #     # features = PCA(n_components=2).fit_transform(stft_data.T)
    #     # plt.scatter(features[:, 0], features[:, 1])
    #     # plt.show()
    # =======
    #     window_size = 10
    #     window_offset = 2
    #     y = np.empty([num_epochs, num_channels, num_frequencies, 10])
    #
    #     i = 0
    #     while i < x.shape[3] - window_size:
    #         y = np.concatenate([y, x[:, :, :, i: i+window_size]], axis=3)
    #         i += window_offset
    #
    #     # stack the epochs together for PCA
    #     if should_visualize:
    #         plot_entropy_of_data_time_and_frequncy_dimensions(pxx, frequencies, times)
    #
    #     # Do PCA on the data in a feature extraction portion
    #     num_components = 32
    #     atures = np.zeros((num_channels, num_components, num_samples))
    #
    #     # do a spectogram of the data
    #     # okay, so basically we gotta decide how to do PCA on this dataset, if we make the dimension of the PCA be frequencies * channels then running PCA
    #     # then PCA will find components for each individual channel, but if we instead have the dimension just be frequencies, then PCA will be finding components
    #     # for the channels at the same time, so the dimension of the eigenvectors will be lower, and we will get less characteristics of each
    #     # channel. tbh idk what is the best to do. intuitively, it would be better for PCA dimension to be frequencies * channels if we had more data.
    #     print(x.shape)
    #     for channel in range(x.shape[0]):
    #         # so this plots the spectogram, it should be:
    #         plt.figure()
    #         plt.imshow(10 * np.log10(np.abs(x)).T, aspect="auto", origin="lower")
    #         plt.title(f"Spectrogram of Channel {channel}")
    #         plt.ylabel("Frequency * Epoch [Hz]")
    #         plt.xlabel("Time [s]")
    #         plt.show()
    # >>>>>>> 0c0e3e5773c816639bd03e1569b2af7206b4f5ab
    #
    #     features = PCA(n_components=2).fit_transform(stft_data.T)
    #
    #     plt.scatter(features[:, 0], features[:, 1])
    #     plt.show()
    #
    #     ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=800)
    #     ica.fit(whitened_raw)
    #     ica.plot_sources(whitened_raw, show_scrollbars=False)
    #     plt.show()
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

    # Define the data paths
    # trial_number = 1
    # subject_id = 103
    # visit_number = 1
    # eeg_data, accel_data, action_data = preprocess(
    #     f"../DataCollection/data/{subject_id}/{visit_number}/{trial_number}/",
    #     actions,
    #     should_visualize=False,
    # )
    # print("Eeg data shape:", eeg_data.shape)
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
