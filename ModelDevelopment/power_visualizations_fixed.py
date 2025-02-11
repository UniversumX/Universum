import preprocessing as pp
import mne
import numpy as np
from scipy.signal import welch
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import os


# ------------------------------------------------------
def compute_power(data, sampling_rate=256, time_interval=0.5):
    """
    Compute the power for each channel using Welch's method.
    
    Parameters:
        data (numpy.ndarray): EEG data of shape (num_epochs, num_channels, frequency_bands, num_samples_per_epoch).
        sampling_rate (int): Sampling rate of the EEG data.
    
    Returns:
        power_map (numpy.ndarray): Power values of shape (num_channels, num_frequencies, num_samples).
    """
    timestamps = data["timestamp"]
    data = data.to_numpy()
    data = np.delete(data, 0, 1)
    num_channels = data.shape[1]
    num_samples = data.shape[0]
    interval = int(time_interval * sampling_rate)
    
    f, first_Pxx = welch(data[0:interval, 0], fs=sampling_rate)
    num_freqs = len(f)
    
    power_per_channel = np.zeros((num_channels, num_freqs, num_samples - interval), dtype=np.complex128)
    
    # Compute power for each channel
    for iteration in range(num_samples - interval):
        start = iteration
        end = interval + iteration
        window = data[start:end, :]
        
        for ch in range(num_channels):
            f, Pxx = welch(window[:, ch], fs=sampling_rate)
            power_per_channel[ch, :, iteration] = Pxx
    
    return f, timestamps, np.abs(power_per_channel)

def get_color_map(num_channels):
    colors = plt.cm.viridis(np.linspace(0, 1, num_channels))
    return colors

def plot_topomap(eeg_data, action_data, electrode, electrode_positions, fps=30, time_interval=0.5, sampling_rate=256):
    """
    Plot the topological map of power values.
    
    Parameters:
        power_values (numpy.ndarray): Power values for each channel.
        electrode_positions (dict): Dictionary of electrode positions {channel: (x, y)}.
    """
    f, timestamps, power_values = compute_power(eeg_data, sampling_rate=sampling_rate, time_interval=time_interval)
    power_values_avg = np.mean(power_values, axis=1)
    num_samples = power_values_avg.shape[1]
    positions = np.array(list(electrode_positions.values()))
    positions = positions * 10
    num_pos = len(positions)

    num_points = 100
    radius = 1.25
    shift = 0.2
    window_length = time_interval * sampling_rate
    action_data = action_data.to_numpy()
    action_idx = [0]
    action_value_map = {action.action_value: name for name, action in actions.items()}
    current_action = ["No action"]

    colors = get_color_map(len(electrode_positions))
    channel_names = list(electrode_positions.keys())

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles) - shift - 0.05))
    positions = np.vstack((positions, circle_points))

    x, y = positions[:, 0], positions[:, 1]

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # Plot
    scalp = Circle((0, radius * (0 - shift)), radius * 1.0, color='black', fill=False, linestyle='--', linewidth=1.5)
    nose = Polygon([(0, radius * (1.12 - shift)), (radius * -0.1, radius * (1.00 - shift)), (radius * 0.1, radius * (1.00 - shift))], color='black', zorder=3)
    left_ear = Line2D([radius * -1.05, radius * -1.15, radius * -1.05], [radius * (0.2 - shift), radius * (0.0 - shift), radius * (-0.2 - shift)], color='black', linewidth=1.5)
    right_ear = Line2D([radius * 1.05, radius * 1.15, radius * 1.05], [radius * (0.2 - shift), radius * (0.0 - shift), radius * (-0.2 - shift)], color='black', linewidth=1.5)
    
    def update(frame):
        nonlocal action_idx, current_action, action_data, action_value_map
        ax.clear()
        ax2.clear()

        ax.add_artist(scalp)
        ax.add_artist(nose)
        ax.add_line(left_ear)
        ax.add_line(right_ear)
        
        # Create grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(min(x), max(x), 100),
            np.linspace(min(y), max(y), 100)
        )
        idx = int(frame*sampling_rate/fps)
        z = power_values_avg[:, idx] # check to see if electrodes alligned with their positions
        z = np.pad(z, (0, num_points), mode='constant')

        grid_z = griddata(positions, z, (grid_x, grid_y), method='cubic')
        
        im = ax.imshow(grid_z, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='viridis')

        # Update scatter plot
        scatter = ax.scatter(x[:num_pos], y[:num_pos], c=z[:num_pos], cmap='viridis', s=250, edgecolor='k')
        
        for label, pos in electrode_positions.items():
            ax.text(pos[0] * 10, pos[1] * 10, label, fontsize=8, ha='center', va='center', color='white')
            
        # Determine action
        timestamp_idx = int(frame * sampling_rate/fps + 0.5*window_length)
        current_time = timestamps[timestamp_idx]
        print(f"Loading: {int(timestamp_idx/num_samples*100) - 1}%  ", end="\r") # Loading bar
        while (action_idx[0] < len(action_data) and current_time >= action_data[action_idx[0], 0]):
            current_action[0] = action_value_map.get(action_data[action_idx[0], 1])
            action_idx[0] += 1

        # Adjust plot appearance
        ax.set_title("Topological Map of EEG Power")
        ax.text(2, 2.2, f"{current_action[0]}", fontsize=14, ha='center', va='center', color='black')
        ax.text(2, -2.6, f"Timestamp: {current_time}", fontsize=13, ha='center', va='center', color='black')
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.4)
        ax.axis('off')
        ax.grid(False)

        #----------------------------------------------
        # PSD plot
        ax2.clear()
        for ch_idx, (channel_name, color) in enumerate(zip(channel_names, colors)):
            graph = ax2.semilogy(f, np.abs(power_values[ch_idx, :, idx]), 
                                color=color, 
                                label=channel_name,
                                alpha=0.7)

        ax2.set_ylim(1e-14, 1e8)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density (µV²/Hz)')
        ax2.set_title('PSD Map All Channels')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        return im, scatter, graph

    ani = FuncAnimation(fig, update, frames=int(num_samples*fps/sampling_rate), interval=1000 / fps, blit=False)
    #plt.show()
    ani.save("eeg_animation_combined.mp4", writer='ffmpeg', fps=fps)
    
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

data_path = f"../DataCollection/data/EEGdata/103/1/1/"
eeg_data_path = os.path.join(data_path, "eeg_data_raw.csv")
action_data_path = os.path.join(data_path, "action_data.csv")
eeg_data = pd.read_csv(eeg_data_path)
action_data = pd.read_csv(action_data_path)

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

sampling_rate = 256

plot_topomap(eeg_data, action_data, 3, electrode_positions, fps=30, time_interval=0.5)