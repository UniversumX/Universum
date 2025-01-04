import preprocessing as pp
import mne
import numpy as np
from scipy.signal import welch
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D


# ------------------------------------------------------
def compute_power(data, sampling_rate):
    """
    Compute the power for each channel using Welch's method.
    
    Parameters:
        data (numpy.ndarray): EEG data of shape (num_epochs, num_channels, frequency_bands, num_samples_per_epoch).
        sampling_rate (int): Sampling rate of the EEG data.
    
    Returns:
        power_map (numpy.ndarray): Power values for each channel of shape (num_channels,).
    """
    num_epochs, num_channels, frequency_bands, num_samples = data.shape
    power_per_channel = np.zeros((num_channels,))
    
    # Average power across epochs
    for channel in range(num_channels):
        channel_data = data[:, channel, :, :].reshape(-1, num_samples)  # Combine epochs
        f, Pxx = welch(channel_data, fs=sampling_rate, axis=-1)  # Power spectral density
        power_per_channel[channel] = np.mean(Pxx)  # Average power
    
    return power_per_channel

def plot_topomap(power_values, electrode_positions):
    """
    Plot the topological map of power values.
    
    Parameters:
        power_values (numpy.ndarray): Power values for each channel.
        electrode_positions (dict): Dictionary of electrode positions {channel: (x, y)}.
    """
    positions = np.array(list(electrode_positions.values()))
    positions = positions * 10
    x, y = positions[:, 0], positions[:, 1]
    z = power_values
    
    # Create grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x) - 0.02, max(x) + 0.02, 100),
        np.linspace(min(y) - 0.02, max(y) + 0.02, 100)
    )
    
    # Interpolate
    grid_z = griddata(positions, z, (grid_x, grid_y), method='cubic')
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_z, extent=(min(x) - 0.02, max(x) + 0.02, min(y) - 0.02, max(y) + 0.02), origin='lower', cmap='viridis')
    plt.scatter(x, y, c=z, cmap='viridis', s=500, edgecolor='k')  # Plot electrode positions

    scalp = Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', linewidth=1.5)
    plt.gca().add_artist(scalp)
    nose = Polygon([(0, 1.02), (-0.1, 0.9), (0.1, 0.9)], color='black', zorder=3)
    plt.gca().add_artist(nose)
    left_ear = Line2D([-1.05, -1.15, -1.05], [0.2, 0.0, -0.2], color='black', linewidth=1.5)
    right_ear = Line2D([1.05, 1.15, 1.05], [0.2, 0.0, -0.2], color='black', linewidth=1.5)
    plt.gca().add_line(left_ear)
    plt.gca().add_line(right_ear)
    
    plt.colorbar(label='Power')
    plt.title('Topological Map of EEG Power')
    plt.show()
    
def get_electrode_positions(electrode_names, montage_name="standard_1020"):
    """
    Get accurate electrode positions for specified electrodes using MNE.

    Parameters:
        electrode_names (list): List of electrode names (e.g., ['CP3', 'C3', 'F5']).
        montage_name (str): Name of the MNE montage (default: 'standard_1020').
    
    Returns:
        dict: Dictionary with electrode names as keys and (x, y) 2D positions as values.
    """
    # Load the montage
    montage = mne.channels.make_standard_montage(montage_name)
    
    # Get the 3D positions of the electrodes
    pos_3d = montage.get_positions()["ch_pos"]
    
    # Filter for the requested electrodes
    electrode_positions = {
        name: (pos_3d[name][0], pos_3d[name][1])  # Keep only x and y
        for name in electrode_names
        if name in pos_3d
    }
    
    return electrode_positions

# ------------------------------------------------------

electrode_names = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
electrode_positions = get_electrode_positions(electrode_names)

eeg_data_path = f"../DataCollection/data/EEGdata/103/1/1/"

from dataclasses import dataclass
@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str
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

data, acell_data, action_data = pp.preprocess(eeg_data_path, actions, False)
sampling_rate = 256
power_values = compute_power(data, sampling_rate)

plot_topomap(power_values, electrode_positions)