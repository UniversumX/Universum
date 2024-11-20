import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_preprocess_data(eeg_file, action_file, transition_window_ms=500):
    """
    Load and preprocess EEG and action data with transition windows
    
    Parameters:
    -----------
    eeg_file: str
        Path to EEG data file
    action_file: str
        Path to action labels file
    transition_window_ms: int
        Milliseconds to exclude before and after action changes (default: 500ms)
    """
    # Load data
    eeg_df = pd.read_csv(eeg_file)
    action_df = pd.read_csv(action_file)
    # print("Columns in action_df:", action_df.columns)
    
    # Convert timestamps to datetime
    eeg_df['timestamp'] = pd.to_datetime(eeg_df['timestamp'])
    action_df['timestamp'] = pd.to_datetime(action_df['timestamp'])
    
    # Sort both dataframes by timestamp
    eeg_df = eeg_df.sort_values('timestamp')
    action_df = action_df.sort_values('timestamp')
    
    # Merge datasets using merge_asof with backward direction
    merged_df = pd.merge_asof(eeg_df,
                             action_df,
                             on='timestamp',
                             direction='backward')
    # print("Columns in merged_df after merge:", merged_df.columns)
    
    # Create a column indicating when actions change
    merged_df['action_changed'] = merged_df['action_value'].shift() != merged_df['action_value']
    
    # Convert transition_window_ms to timedelta
    window = pd.Timedelta(milliseconds=transition_window_ms)
    
    # Find timestamps where actions change
    change_timestamps = merged_df.loc[merged_df['action_changed'], 'timestamp']
    
    # Create a mask for data points that are far enough from transitions
    mask = pd.Series(True, index=merged_df.index)
    
    for change_time in change_timestamps:
        # Mask out data within window before and after each transition
        mask &= ~((merged_df['timestamp'] > change_time - window) & 
                 (merged_df['timestamp'] < change_time + window))
    
    # Apply the mask to keep only steady-state periods
    steady_state_df = merged_df[mask].copy()
    
    # Drop the helper column
    steady_state_df = steady_state_df.drop('action_changed', axis=1)
    
    return steady_state_df

def combine_datasets(file_pairs, transition_window_ms=500):
    """
    Combine data from multiple file pairs with transition windows
    """
    all_data = []
    for eeg_file, action_file in file_pairs:
        merged_df = load_and_preprocess_data(eeg_file, action_file, 
                                           transition_window_ms=transition_window_ms)
        all_data.append(merged_df)
        
    # Concatenate all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def compute_and_plot_psd(combined_data, channels, fs=250, nperseg=1024):
    # print('combined_data', combined_data)
    """
    Compute and plot PSD for all channels and all actions
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Define colors and labels for each action
    action_colors = {
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'purple'
    }
    
    for idx, channel in enumerate(channels):
        # Process each action
        for action in range(1, 5):
            # Get data for this action
            action_data = combined_data[combined_data['action_value'] == action][channel].values
            
            if len(action_data) > 0:  # Only process if we have data for this action
                # Compute PSD
                f, pxx = signal.welch(action_data, fs=fs, nperseg=nperseg)
                
                # Plot
                axes[idx].semilogy(f, pxx, color=action_colors[action], 
                                 linewidth=2, label=f'Action {action}',
                                 alpha=0.7)  # Added some transparency for better visibility
        
        axes[idx].grid(True)
        axes[idx].set_xlim(0, 45)
        axes[idx].set_ylim(1, 10000000)
        axes[idx].set_title(f'Channel {channel}')
        axes[idx].set_xlabel('Frequency (Hz)')
        axes[idx].set_ylabel('Power Spectral Density')
        axes[idx].legend()
    
    plt.suptitle('Power Spectral Density Analysis by Channel\nAll Actions Combined Across Recordings')
    plt.tight_layout()
    plt.show()
    
    # Print statistics about the data
    print("\nData points per action:")
    for action in range(1, 5):
        count = len(combined_data[combined_data['action_value'] == action])
        print(f"Action {action}: {count} points")

def compute_and_plot_channel_avg_psd(combined_data, channels, standardize, fs=250, nperseg=1024):
    """
    Compute and plot a single average PSD value per action for each channel.
    """
    # Define colors and labels for each action
    action_colors = {
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'purple'
    }
    
    # Dictionary to store the average PSD value per action for each channel
    channel_avg_psd = {channel: {action: 0 for action in range(1, 5)} for channel in channels}
    
    # Loop over channels
    for channel in channels:
        # Process each action
        for action in range(1, 5):
            # Get data for this action
            action_data = combined_data[combined_data['action_value'] == action][channel].values
            
            if len(action_data) > 0:  # Only process if we have data for this action
                # Compute PSD
                f, pxx = signal.welch(action_data, fs=fs, nperseg=nperseg)
                
                # Calculate the average PSD for this action by averaging across frequencies
                avg_psd_value = np.mean(pxx)
                
                # Store this average PSD value in the dictionary
                channel_avg_psd[channel][action] = avg_psd_value
    
    # Plot the results
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, channel in enumerate(channels):
        actions = list(channel_avg_psd[channel].keys())
        avg_psd_values = list(channel_avg_psd[channel].values())
        
        # Create a bar plot for each channel
        axes[idx].bar(actions, avg_psd_values, color=[action_colors[action] for action in actions])
        axes[idx].set_title(f'Channel {channel}')
        axes[idx].set_xlabel('Action')
        axes[idx].set_ylabel('Average Power Spectral Density')
        if standardize:
            axes[idx].set_ylim(1,11000)
        axes[idx].set_xticks(actions)
        axes[idx].set_xticklabels([f'Action {action}' for action in actions])
    
    plt.suptitle('Average PSD per Action for Each Channel')
    plt.tight_layout()
    plt.show()
    
    # Print the average PSD values per action for each channel
    print("\nAverage PSD value per action for each channel:")
    for channel in channels:
        print(f"\nChannel {channel}:")
        for action, avg_psd in channel_avg_psd[channel].items():
            print(f"  Action {action}: {avg_psd:.2f}")

def compute_total_power_per_action(combined_data, channels, fs=250, nperseg=1024):
    """
    Compute total power across all frequencies for each action and channel
    
    Parameters:
    -----------
    combined_data: pandas DataFrame
        Combined EEG and action data
    channels: list
        List of channel names
    fs: int
        Sampling frequency (default: 250 Hz)
    nperseg: int
        Length of each segment for PSD calculation (default: 1024)
        
    Returns:
    --------
    dict: Channel-wise total power for each action
    dict: Channel-wise power distributions (frequencies and PSD values)
    """
    # Dictionary to store total power per action for each channel
    channel_total_power = {channel: {action: 0 for action in range(1, 5)} 
                          for channel in channels}
    
    # Dictionary to store power distributions (for potential detailed analysis)
    power_distributions = {channel: {action: {'freqs': None, 'psd': None} 
                          for action in range(1, 5)} for channel in channels}
    
    # Calculate power for each channel and action
    for channel in channels:
        for action in range(1, 5):
            # Get data for this action
            action_data = combined_data[combined_data['action_value'] == action][channel].values
            
            if len(action_data) > 0:
                # Compute PSD
                freqs, pxx = signal.welch(action_data, fs=fs, nperseg=nperseg)
                
                # Calculate total power by integrating over all frequencies
                # Using trapezoidal integration for better accuracy
                total_power = np.trapz(pxx, freqs)
                
                # Store results
                channel_total_power[channel][action] = total_power
                power_distributions[channel][action] = {
                    'freqs': freqs,
                    'psd': pxx
                }
    
    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Colors for different actions
    action_colors = {
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'purple'
    }
    
    # Create bar plots for each channel
    for idx, channel in enumerate(channels):
        actions = list(channel_total_power[channel].keys())
        powers = list(channel_total_power[channel].values())
        
        # Create bar plot
        bars = axes[idx].bar(actions, powers, 
                           color=[action_colors[action] for action in actions])
        axes[idx].set_title(f'Channel {channel}')
        axes[idx].set_xlabel('Action')
        axes[idx].set_ylabel('Total Power')
        axes[idx].set_xticks(actions)
        axes[idx].set_xticklabels([f'Action {action}' for action in actions])
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2e}',
                         ha='center', va='bottom', rotation=0)
    
    plt.suptitle('Total Power per Action for Each Channel')
    plt.tight_layout()
    plt.show()
    
    # Print numerical results
    print("\nTotal power per action for each channel:")
    for channel in channels:
        print(f"\nChannel {channel}:")
        for action, power in channel_total_power[channel].items():
            print(f"  Action {action}: {power:.2e}")
            
    return channel_total_power, power_distributions
def compute_power_metrics_per_action(combined_data, channels, fs=250, nperseg=1024):
    """
    Compute both total and average power across frequencies for each action and channel
    with standardized y-axis ranges for easier comparison
    """
    # Dictionaries to store power metrics per action for each channel
    channel_total_power = {channel: {action: 0 for action in range(1, 5)} 
                          for channel in channels}
    channel_avg_power = {channel: {action: 0 for action in range(1, 5)} 
                        for channel in channels}
    
    # Calculate power metrics for each channel and action
    for channel in channels:
        for action in range(1, 5):
            action_data = combined_data[combined_data['action_value'] == action][channel].values
            
            if len(action_data) > 0:
                freqs, pxx = signal.welch(action_data, fs=fs, nperseg=nperseg)
                total_power = np.trapz(pxx, freqs)
                # Calculate average power by dividing total power by frequency range
                avg_power = total_power / (freqs[-1] - freqs[0])
                
                channel_total_power[channel][action] = total_power
                channel_avg_power[channel][action] = avg_power
    
    # Find global min and max for both metrics to set y-axis limits
    all_total_powers = [power for channel_dict in channel_total_power.values() 
                       for power in channel_dict.values()]
    all_avg_powers = [power for channel_dict in channel_avg_power.values() 
                     for power in channel_dict.values()]
    
    total_power_max = max(all_total_powers)
    total_power_min = min(all_total_powers)
    avg_power_max = max(all_avg_powers)
    avg_power_min = min(all_avg_powers)
    
    # Create subplots
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    action_colors = {
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'purple'
    }
    
    # Plot total power (top two rows)
    # for idx, channel in enumerate(channels):
    #     ax = axes[idx // 4, idx % 4]
    #     actions = list(channel_total_power[channel].keys())
    #     powers = list(channel_total_power[channel].values())
        
    #     bars = ax.bar(actions, powers, 
    #                  color=[action_colors[action] for action in actions])
    #     ax.set_title(f'Channel {channel} - Total Power')
    #     ax.set_xlabel('Action')
    #     ax.set_ylabel('Total Power')
    #     ax.set_xticks(actions)
    #     ax.set_xticklabels([f'Action {action}' for action in actions])
        
    #     # Set y-axis limits with a little padding (10%)
    #     padding = (total_power_max - total_power_min) * 0.1
    #     ax.set_ylim(total_power_min - padding, total_power_max + padding)
        
        # Add value labels
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(bar.get_x() + bar.get_width()/2., height,
        #            f'{height:.2e}',
        #            ha='center', va='bottom', rotation=45)
    
    # Plot average power (bottom two rows)
    for idx, channel in enumerate(channels):
        ax = axes[idx // 4 + 2, idx % 4]
        actions = list(channel_avg_power[channel].keys())
        powers = list(channel_avg_power[channel].values())
        
        bars = ax.bar(actions, powers,
                     color=[action_colors[action] for action in actions])
        ax.set_title(f'Channel {channel} - Average Power')
        ax.set_xlabel('Action')
        ax.set_ylabel('Average Power')
        ax.set_xticks(actions)
        ax.set_xticklabels([f'Action {action}' for action in actions])
        
        # Set y-axis limits with a little padding (10%)
        padding = (avg_power_max - avg_power_min) * 0.1
        ax.set_ylim(avg_power_min - padding, avg_power_max + padding)
        
        # Add value labels
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(bar.get_x() + bar.get_width()/2., height,
        #            f'{height:.2e}',
        #            ha='center', va='bottom', rotation=45)
    
    plt.suptitle('Total vs Average Power per Action for Each Channel\n(Standardized Y-axes)')
    plt.tight_layout()
    plt.show()
    
    # Print numerical results
    print("\nPower metrics per action for each channel:")
    for channel in channels:
        print(f"\nChannel {channel}:")
        print("Total Power:")
        for action, power in channel_total_power[channel].items():
            print(f"  Action {action}: {power:.2e}")
        print("Average Power:")
        for action, power in channel_avg_power[channel].items():
            print(f"  Action {action}: {power:.2e}")
            
    # Print the global ranges
    print("\nGlobal Ranges:")
    print(f"Total Power: {total_power_min:.2e} to {total_power_max:.2e}")
    print(f"Average Power: {avg_power_min:.2e} to {avg_power_max:.2e}")
    
    # Calculate and print the ratio between total and average power
    print("\nRatio between Total and Average Power:")
    for channel in channels:
        print(f"\nChannel {channel}:")
        for action in range(1, 5):
            total = channel_total_power[channel][action]
            avg = channel_avg_power[channel][action]
            if avg != 0:
                ratio = total / avg
                print(f"  Action {action}: {ratio:.2f}")

    return channel_total_power, channel_avg_power

def main():
    # Define file pairs
    file_pairs = [
        ('../DataCollection/data/105/1/1/eeg_data_raw.csv', 
         '../DataCollection/data/105/1/1/action_data.csv'),
        ('../DataCollection/data/105/1/2/eeg_data_raw.csv', 
         '../DataCollection/data/105/1/2/action_data.csv'),
        ('../DataCollection/data/105/1/5/eeg_data_raw.csv', 
         '../DataCollection/data/105/1/5/action_data.csv'),
         ('../DataCollection/data/105/1/6/eeg_data_raw.csv', 
         '../DataCollection/data/105/1/6/action_data.csv'),
         ('../DataCollection/data/105/1/7/eeg_data_raw.csv', 
         '../DataCollection/data/105/1/7/action_data.csv'),
    ]
    
    # file_pairs = [
    #     ('../DataCollection/data/104/1/1/eeg_data_raw.csv', 
    #      '../DataCollection/data/104/1/1/action_data.csv'),
    #     ('../DataCollection/data/104/1/2/eeg_data_raw.csv', 
    #      '../DataCollection/data/104/1/2/action_data.csv'),
    #     ('../DataCollection/data/104/1/3/eeg_data_raw.csv', 
    #      '../DataCollection/data/104/1/3/action_data.csv'),
    # ]

    # file_pairs = [
    #     ('../DataCollection/data/103/1/1/eeg_data_raw.csv', 
    #      '../DataCollection/data/103/1/1/action_data.csv'),
    #     ('../DataCollection/data/103/1/4/eeg_data_raw.csv', 
    #      '../DataCollection/data/103/1/4/action_data.csv'),
    #     ('../DataCollection/data/103/1/5/eeg_data_raw.csv', 
    #      '../DataCollection/data/103/1/5/action_data.csv'),
    # ]
    # Define channels
    channels = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    
    # Combine all datasets
    print("Loading and combining datasets...")
    combined_data = combine_datasets(file_pairs)
    
    # Print some statistics
    total_points = len(combined_data)
    print(f"\nTotal number of datapoints: {total_points}")
    print(f"Datapoints per recording: {[len(load_and_preprocess_data(eeg, action)) for eeg, action in file_pairs]}")
    
    # Compute and plot PSD
    # print("\nComputing and plotting PSD...")
    compute_and_plot_psd(combined_data, channels)

    compute_and_plot_channel_avg_psd(combined_data, channels, False)
    compute_and_plot_channel_avg_psd(combined_data, channels, True)

    # In your main function:
    # total_power, distributions = compute_total_power_per_action(combined_data, channels)

    # total_power, avg_power = compute_power_metrics_per_action(combined_data, channels)

if __name__ == "__main__":
    main()