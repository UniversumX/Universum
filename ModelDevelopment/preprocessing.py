import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mne
import pandas as pd
from scipy import stats
from scipy import integrate

matplotlib.use("TkAgg")

# Load the data
trial = 2
eeg_data_path = f"../DataCollection/data/3/1/{trial}/eeg_data_raw.csv"
accel_data_path = f"../DataCollection/data/3/1/{trial}/accelerometer_data.csv"

eeg_data = pd.read_csv(eeg_data_path)
accel_data = pd.read_csv(accel_data_path)

# Convert timestamp to time since last epoch
accel_data["timestamp"] = pd.to_datetime(accel_data["timestamp"]).astype(int) / 10**9
eeg_data["timestamp"] = pd.to_datetime(eeg_data["timestamp"]).astype(int) / 10**9

# Make the timestamps aligned/overlapping
prev_eeg_shape = eeg_data.shape
eeg_data = eeg_data[
    (eeg_data["timestamp"] >= accel_data["timestamp"].iloc[0])
    & (eeg_data["timestamp"] <= accel_data["timestamp"].iloc[-1])
]
accel_data = accel_data[
    (accel_data["timestamp"] >= eeg_data["timestamp"].iloc[0])
    & (accel_data["timestamp"] <= eeg_data["timestamp"].iloc[-1])
]

new_eeg_shape = eeg_data.shape
print(f"Trimmed EEG data from {prev_eeg_shape} to {new_eeg_shape}")
accel_data = accel_data.drop(columns=["device_id"])

print(eeg_data.head())
print(accel_data.head())

# Align accel data with eeg data
accel_data = accel_data.astype(float)
eeg_data = eeg_data.astype(float)


def time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data):
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


accel_data = time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data)
print(accel_data.head())

sampling_freq = 1 / eeg_data["timestamp"].diff().mean()
print(f"Sampling frequency: {sampling_freq:.2f} Hz")

sfreq = 256
ch_names = eeg_data.columns[1:].tolist()
ch_types = ["eeg"] * len(ch_names)

head_tilt = np.where(accel_data["roll"] > -45, 1, 0)

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
eeg_data_array = eeg_data[ch_names].to_numpy().T
raw = mne.io.RawArray(eeg_data_array, info)
cutoff_max = 30  # Hz
cutoff_min = 1
raw.filter(l_freq=cutoff_min, h_freq=cutoff_max, fir_design="firwin")

changes = np.diff(head_tilt)
events = np.where(changes != 0)[0] + 1
events = np.column_stack((events, np.zeros(len(events), int), head_tilt[events]))

print(f"Events: {events}")


def differential_entropy(pdf, lower, upper):
    def integrand(x):
        p = pdf(x)
        return -p * np.log2(p) if p > 0 else 0

    result, _ = integrate.quad(integrand, lower, upper)
    return result


NFFT = 1024
percent_overlap = 1 - (1 / 32)

# 4.65
# 9.36
# 31.982
get_rid_of_these_frequencies = [4.65, 9.36, 31.982]
raw.notch_filter(freqs=get_rid_of_these_frequencies)


def plot_fft(data, sfreq, nfft, percent_overlap):
    pxx, freqs, bins, im = plt.specgram(
        data, Fs=sfreq, NFFT=nfft, noverlap=int(percent_overlap * nfft)
    )
    plt.show()


def plot_fft_with_entropy(data, sfreq, nfft, percent_overlap):
    pxx, freqs, bins, im = plt.specgram(
        data, Fs=sfreq, NFFT=nfft, noverlap=int(percent_overlap * nfft)
    )

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


# plot_fft(raw.get_data()[0], sfreq, NFFT, percent_overlap)

fig = raw.compute_psd(tmax=np.inf, fmax=sfreq // 2).plot(
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


# fig = whitened_raw.compute_psd(tmax=np.inf, fmax=sfreq // 2).plot(
#     average=False, amplitude=False, picks="data", exclude="bads"
# )
# plt.title("Whitened Data")
# plt.show()

# Check the covariance matrix

whitened_noise_cov = np.cov(whitened_data)
# plt.matshow(whitened_noise_cov)
# plt.show()

plot_fft_with_entropy(whitened_raw.get_data()[0], sfreq, NFFT, percent_overlap)

print("Performing ICA...")
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
explained_var_ratio = ica.get_explained_variance_ratio(whitened_raw)
print(repr(ica))
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: {ratio:.2f}"
    )

explained_var_ratio = ica.get_explained_variance_ratio(
    whitened_raw, components=[0], ch_type="eeg"
)
ratio_percent = round(100 * explained_var_ratio["eeg"])
print(
    f"Fraction of variance in EEG signal explained by first component: {ratio_percent}%"
)

sources = ica.get_sources(whitened_raw).get_data()
first_component_signal = sources[0, :]

plt.figure()
plt.plot(first_component_signal)
plt.title("First Component Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()

print("First component signal extracted.")

plot_fft(first_component_signal, sfreq, NFFT, percent_overlap)
