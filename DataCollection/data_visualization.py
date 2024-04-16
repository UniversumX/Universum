import pandas as pd
import matplotlib.pyplot as plt

# Load the data
def load_data(filename):
    return pd.read_csv(filename)

# Plot EEG data using row numbers
def plot_data(df):
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Assuming you want to plot data for all channels, or you can filter by a specific channel
    channels = df['channel'].unique()
    
    for channel in channels:
        # Filter data for one channel at a time
        channel_data = df[df['channel'] == channel]
        
        # Plot using row numbers as the x-axis
        plt.plot(channel_data.index, channel_data['value'], label=f'Channel {channel}')

    plt.title('EEG Data Over Rows')
    plt.xlabel('Row Number')
    plt.ylabel('EEG Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Assuming your CSV file is named 'eeg_data.csv'
df = load_data('eeg_data.csv')
plot_data(df)
