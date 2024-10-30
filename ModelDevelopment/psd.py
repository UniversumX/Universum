import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_preprocess_data(eeg_file, action_file):
    """
    Load and preprocess EEG and action data
    """
    # Load data
    eeg_df = pd.read_csv(eeg_file)
    action_df = pd.read_csv(action_file)
    
    # Convert timestamps to datetime
    eeg_df['timestamp'] = pd.to_datetime(eeg_df['timestamp'])
    action_df['timestamp'] = pd.to_datetime(action_df['timestamp'])
    
    # Merge datasets on nearest timestamp
    merged_df = pd.merge_asof(eeg_df.sort_values('timestamp'), 
                             action_df.sort_values('timestamp'),
                             on='timestamp',
                             direction='nearest')
    
    return merged_df

def compute_psd_by_action(data, channel, fs=250, nperseg=1024):
    """
    Compute PSD for each action value for a given channel
    """
    psd_dict = {}
    unique_actions = sorted(data['action_value'].unique())
    
    for action in unique_actions:
        signal_data = data[data['action_value'] == action][channel].values
        f, pxx = signal.welch(signal_data, fs=fs, nperseg=nperseg)
        psd_dict[action] = (f, pxx)
    
    return psd_dict

def plot_psd_comparison(psd_dict, channel_name):
    """
    Plot PSD comparison for different actions
    """
    plt.figure(figsize=(12, 6))
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for idx, (action, (f, pxx)) in enumerate(psd_dict.items()):
        plt.semilogy(f, pxx, colors[idx % len(colors)], 
                    label=f'Action {action}')
    
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'PSD Comparison by Action Value - Channel {channel_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def analyze_all_channels(data, channels, fs=250, nperseg=1024):
    """
    Analyze PSD for all channels
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for idx, channel in enumerate(channels):
        psd_dict = compute_psd_by_action(data, channel, fs, nperseg)
        
        for action_idx, (action, (f, pxx)) in enumerate(psd_dict.items()):
            axes[idx].semilogy(f, pxx, colors[action_idx % len(colors)], 
                             label=f'Action {action}')
        
        axes[idx].grid(True)
        axes[idx].set_title(f'Channel {channel}')
        axes[idx].set_xlabel('Frequency (Hz)')
        axes[idx].set_ylabel('PSD')
    
    plt.suptitle('Power Spectral Density Analysis by Channel and Action')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
merged_data = load_and_preprocess_data('../DataCollection/data/103/1/1/eeg_data_raw.csv', '../DataCollection/data/103/1/1/action_data.csv')
channels = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
analyze_all_channels(merged_data, channels)

# # for individual channel
# channel = 'CP3'
# psd_dict = compute_psd_by_action(merged_data, channel)
# plot_psd_comparison(psd_dict, channel)