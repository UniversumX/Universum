"""Example program to show how to read a multi-channel time series from LSL."""
# https://support.neurosity.co/hc/en-us/articles/360039387812-Reading-data-into-Python-via-LSL
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import time
import csv

from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import csv
import time

def write_data_to_csv(timestamp, sample):
    with open('eeg_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp] + list(sample))

def collect_eeg_data():
    try:
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        print("Connected to EEG stream. Collecting data...")

        while True:
            sample, timestamp = inlet.pull_sample()
            timestamp_str = datetime.fromtimestamp(timestamp).strftime("%F %T.%f")[:-3]
            write_data_to_csv(timestamp_str, sample)

    except Exception as e:
        print(f"Error: {e}. Attempting to reconnect...")
        time.sleep(2)  # Wait before trying to reconnect

if __name__ == "__main__":
    collect_eeg_data()