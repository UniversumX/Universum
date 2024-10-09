import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mne
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, FastICA
import pywt

# Set up matplotlib if needed on specific environments
matplotlib.use("TkAgg")

# Function to list all data files in the DataCollection folder
def list_all_data_files(root_folder):
    eeg_files, accel_files = [], []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith("eeg_data_raw.csv"):
                eeg_files.append(os.path.join(subdir, file))
            elif file.endswith("accelerometer_data.csv"):
                accel_files.append(os.path.join(subdir, file))
    return eeg_files, accel_files

# Function to plot the FFT (Spectrogram) of the signal
def plot_fft(data, sampling_frequency, fft_window_size, percent_overlap_between_windows, ax):
    """
    Plots the FFT (spectrogram) of the data on the provided axes.
    """
    pxx, freqs, bins, im = ax.specgram(
        data,
        Fs=sampling_frequency,
        NFFT=fft_window_size,
        noverlap=int(percent_overlap_between_windows * fft_window_size),
    )
    ax.set_title("Spectrogram of the EEG Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax, label='Intensity (dB)')
    return pxx, freqs, bins

# Function to remove noise using ICA/PCA
def perform_ica_pca(raw, method='ICA', n_components=8):
    if method == 'ICA':
        ica = FastICA(n_components=n_components, max_iter=800, random_state=97)
        sources = ica.fit_transform(raw.T)  # Transposing to (n_samples, n_channels)
        return sources, ica.mixing_, ica
    elif method == 'PCA':
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(raw.T)  # Transposing for PCA
        return components, pca.explained_variance_ratio_, pca

# Function to plot the Power Spectral Density (PSD) of the EEG data
def plot_psd(raw, ax):
    """
    Computes the Power Spectral Density (PSD) of the raw data and plots it on the provided axis.
    """
    psds = raw.compute_psd(fmax=120)  # Compute PSD
    psd_values, freqs = psds.get_data(return_freqs=True)  # Get PSD and frequencies

    # Convert power to dB and plot
    psds_db = 10 * np.log10(psd_values)
    ax.plot(freqs, psds_db.T, color='k', alpha=0.5)
    ax.set_title('Power Spectral Density (PSD)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB µV²/Hz)')
    ax.set_xlim([0, 120])

# Function to perform wavelet denoising
def wavelet_denoise(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet)

# Main preprocessing function
def preprocess_data(eeg_file, accel_file=None):
    eeg_data = pd.read_csv(eeg_file)

    # Convert timestamps and clean unnecessary columns
    eeg_data["timestamp"] = pd.to_datetime(eeg_data["timestamp"]).astype(int) / 10**9
    eeg_data.drop(columns=["device_id"], inplace=True, errors='ignore')

    # Prepare MNE RawArray
    sampling_frequency = 256
    ch_names = eeg_data.columns[1:].tolist()
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_frequency, ch_types=ch_types)
    eeg_data_array = eeg_data[ch_names].to_numpy().T  # Ensuring 2D (n_channels, n_times)
    raw = mne.io.RawArray(eeg_data_array, info)

    # Apply bandpass filter and remove noise
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin")

    # Perform ICA for noise removal
    sources_ica, _, ica_model = perform_ica_pca(raw.get_data(), method='ICA', n_components=8)  # Passing 2D array

    # Perform wavelet denoising on ICA sources
    denoised_data = wavelet_denoise(sources_ica[0])

    # Create a figure with subplots for both Spectrogram and PSD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the Power Spectral Density (PSD) on ax1
    plot_psd(raw, ax1)

    # Plot the spectrogram on ax2
    plot_fft(raw.get_data()[0], sampling_frequency, 1024, 0.95, ax2)

    # Adjust layout and show both plots in a single window
    plt.tight_layout()
    plt.show()

    return denoised_data

# Process all data
def process_all_data(data_folder):
    eeg_files, _ = list_all_data_files(data_folder)
    for eeg_file in eeg_files:
        denoised_data = preprocess_data(eeg_file)
        print(f"Processed file: {eeg_file}")

# Run preprocessing
if __name__ == "__main__":
    data_folder = "../DataCollection"
    process_all_data(data_folder)
