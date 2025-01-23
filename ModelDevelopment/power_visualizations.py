import preprocessing as pp
import mne
import numpy as np
from scipy.signal import welch
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation


# ------------------------------------------------------
def compute_power(data, sampling_rate):
    """
    Compute the power for each channel using Welch's method.
    
    Parameters:
        data (numpy.ndarray): EEG data of shape (num_epochs, num_channels, frequency_bands, num_samples_per_epoch).
        sampling_rate (int): Sampling rate of the EEG data.
    
    Returns:
        power_map (numpy.ndarray): Power values of shape (num_channels, num_frequencies, num_samples).
    """
    num_epochs, num_channels, frequency_bands, num_samples = data.shape
    
    # Get number of frequencies from first run of welch
    channel_data = data[:, 0, :, :].reshape(-1, num_samples)
    f, Pxx = welch(channel_data, fs=sampling_rate, axis=-1)
    num_frequencies = Pxx.shape[0]  # Get actual number of frequencies
    
    # Initialize output array with correct shape from Pxx dimensions
    power_per_channel = np.zeros((num_channels, num_frequencies, num_samples))
    
    # Compute power for each channel
    for channel in range(num_channels):
        channel_data = data[:, channel, :, :].reshape(-1, num_samples)
        f, Pxx = welch(channel_data, fs=sampling_rate, axis=-1)
        power_per_channel[channel] = Pxx
    
    return power_per_channel

def plot_topomap(power_values, electrode_positions, frequency_idx, fps=10):
    """
    Plot the topological map of power values.
    
    Parameters:
        power_values (numpy.ndarray): Power values for each channel.
        electrode_positions (dict): Dictionary of electrode positions {channel: (x, y)}.
    """
    num_samples = data.shape[3]
    positions = np.array(list(electrode_positions.values()))
    positions = positions * 10
    x, y = positions[:, 0], positions[:, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot
    shift = 0.27
    scale = 1.1
    scalp = Circle((0, scale * (0 - shift)), scale * 1.0, color='black', fill=False, linestyle='--', linewidth=1.5)
    nose = Polygon([(0, scale * (1.12 - shift)), (scale * -0.1, scale * (1.00 - shift)), (scale * 0.1, scale * (1.00 - shift))], color='black', zorder=3)
    
    left_ear = Line2D([scale * -1.05, scale * -1.15, scale * -1.05], [scale * (0.2 - shift), scale * (0.0 - shift), scale * (-0.2 - shift)], color='black', linewidth=1.5)
    right_ear = Line2D([scale * 1.05, scale * 1.15, scale * 1.05], [scale * (0.2 - shift), scale * (0.0 - shift), scale * (-0.2 - shift)], color='black', linewidth=1.5)
    
    def update(frame):
        ax.clear()

        ax.add_artist(scalp)
        ax.add_artist(nose)
        ax.add_line(left_ear)
        ax.add_line(right_ear)
        
        # Create grid
        edge_size = 0.4
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(x) - edge_size, max(x) + edge_size, 100),
            np.linspace(min(y) - edge_size, max(y) + edge_size, 100)
        )
        z = power_values[:, frequency_idx, frame]
        grid_z = griddata(positions, z, (grid_x, grid_y), method='cubic')
        
        im = ax.imshow(grid_z, extent=(min(x) - edge_size, max(x) + edge_size, min(y) - edge_size, max(y) + edge_size), origin='lower', cmap='viridis')

        # Update scatter plot
        scatter = ax.scatter(x, y, c=z, cmap='viridis', s=200, edgecolor='k')
        
        for label, pos in electrode_positions.items():
            ax.text(pos[0] * 10, pos[1] * 10, label, fontsize=8, ha='center', va='center', color='white')

        # Adjust plot appearance
        ax.set_title(f'Topological Map - Timestamp {frame}')
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.grid(False)

        return im, scatter

    ani = FuncAnimation(fig, update, frames=num_samples, interval=1000 / fps, blit=False)
    plt.show()
    ani.save("eeg_animation.gif", writer='Pillow', fps=fps)
    
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

print(power_values.shape)

plot_topomap(power_values, electrode_positions, 2)