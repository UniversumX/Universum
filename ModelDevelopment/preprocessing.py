import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mne
import pandas as pd
from scipy import stats
from scipy import integrate

# I (Matt) am running nixos and if I don't set this command then matplotlib won't show
matplotlib.use("TkAgg")


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


def plot_entropy_of_data_time_and_frequncy_dimensions(pxx, freqs, bins):
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
    plt.plot(bins, frame_entropies)
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


# Define the data paths
trial = 2
eeg_data_path = f"../DataCollection/data/3/1/{trial}/eeg_data_raw.csv"
accel_data_path = f"../DataCollection/data/3/1/{trial}/accelerometer_data.csv"

# Load data as CSV
eeg_data = pd.read_csv(eeg_data_path)
accel_data = pd.read_csv(accel_data_path)

# Convert timestamp to time since last epoch (a float)
accel_data["timestamp"] = pd.to_datetime(accel_data["timestamp"]).astype(int) / 10**9
eeg_data["timestamp"] = pd.to_datetime(eeg_data["timestamp"]).astype(int) / 10**9

# Get rid of device id as we don't care about it
accel_data = accel_data.drop(columns=["device_id"])

# Make the timestamps so that they start and end at the same time, throw out data outside the starting/stopping times of each dataset
eeg_data = eeg_data[
    (eeg_data["timestamp"] >= accel_data["timestamp"].iloc[0])
    & (eeg_data["timestamp"] <= accel_data["timestamp"].iloc[-1])
]
accel_data = accel_data[
    (accel_data["timestamp"] >= eeg_data["timestamp"].iloc[0])
    & (accel_data["timestamp"] <= eeg_data["timestamp"].iloc[-1])
]


# Make the data floats (tbh idt we need this)
accel_data = accel_data.astype(float)
eeg_data = eeg_data.astype(float)

# Time align the data by linearly interpolating the accelerometer data
accel_data = time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data)

# Make sure the sampling frequency is the sampling frequency said on the device
# sampling_freq = 1 / eeg_data["timestamp"].diff().mean()
# print(f"Sampling frequency: {sampling_freq:.2f} Hz")

sampling_frequency = 256

# Create column names (mne Raw Array needs this)
ch_names = eeg_data.columns[1:].tolist()
ch_types = ["eeg"] * len(ch_names)

### TODO:
# Read the action_data.csv and use mne events to label specific events in the data, incorporate that with the `raw` variable
# events =

info = mne.create_info(ch_names=ch_names, sfreq=sampling_frequency, ch_types=ch_types)
eeg_data_array = eeg_data[ch_names].to_numpy().T
raw = mne.io.RawArray(eeg_data_array, info)


cutoff_max = 45  # Cut off frequency for band filter
cutoff_min = 1  # Cut off frequency for band filter
raw.filter(l_freq=cutoff_min, h_freq=cutoff_max, fir_design="firwin")


fft_window_size = 1024
percent_overlap = 1 - (1 / 32)

# These frequencies seem to be noise
# 4.65
# 9.36
# 31.982
get_rid_of_these_frequencies = [4.65, 9.36, 31.982]
raw.notch_filter(freqs=get_rid_of_these_frequencies)


fig = raw.compute_psd(tmax=np.inf, fmax=sampling_frequency // 2).plot(
    average=False, amplitude=False, picks="data", exclude="bads"
)
plt.show()
# whiten the data with PCA

# Whiten the data :)
noise_cov = mne.compute_raw_covariance(raw).data
# find the eigenvectors of the covariance matrix to do PCA
epsilon = 1e-8
eigenvalues, eigenvectors = np.linalg.eig(noise_cov)
inverse_lambda = np.diag(1 / (eigenvalues + epsilon))
whitened_data = np.sqrt(inverse_lambda) @ eigenvectors.T @ raw.get_data()
whitened_raw = mne.io.RawArray(whitened_data, raw.info)


# DO some plotting
pxx, freqs, bins = plot_fft(
    whitened_raw.get_data()[0], sampling_frequency, fft_window_size, percent_overlap
)
plot_entropy_of_data_time_and_frequncy_dimensions(pxx, freqs, bins)

# Perform ICA on the data
ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=800)
ica.fit(whitened_raw)
ica.plot_sources(whitened_raw, show_scrollbars=False)
plt.show()


def print_relative_importance_of_ICA_features(ica):
    for i, component in enumerate(ica.mixing_matrix_):
        explained_var_ratio = ica.get_explained_variance_ratio(
            whitened_raw, components=[i], ch_type="eeg"
        )
        print(
            f"Fraction of variance in EEG signal explained by {i}th component: {explained_var_ratio['eeg']}"
        )


print_relative_importance_of_ICA_features(ica)

sources = ica.get_sources(whitened_raw).get_data()
first_component_signal = sources[0, :]

# Ignore this for now: vvvvv
from sklearn.decomposition import NMF


# After whitening the data
print("Performing NMF...")
n_components = 8  # You can adjust this number

print("hiii")

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
