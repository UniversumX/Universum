import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


# Load the data
def load_data(filename):
    return pd.read_csv(filename)


# Plot EEG data using timestamps
def plot_data(df):
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Get the list of channels, ignoring the 'timestamp' column
    channels = df.columns[1:]  # Exclude 'timestamp'

    # Plot data for each channel
    for channel in channels:
        plt.plot(df["timestamp"], df[channel], label=f"Channel {channel}")

    plt.title("EEG Data Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("EEG Value")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Ensure the layout fits well
    plt.show()


# Assuming your CSV file is named 'eeg_data.csv'
df = load_data("eeg_data.csv")

# Convert 'timestamp' to a datetime object for better plotting
df["timestamp"] = pd.to_datetime(df["timestamp"])

plot_data(df)
