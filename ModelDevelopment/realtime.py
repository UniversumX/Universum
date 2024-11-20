"""Example program to show how to read a multi-channel time series from LSL."""
# https://support.neurosity.co/hc/en-us/articles/360039387812-Reading-data-into-Python-via-LSL



from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import csv
import time


def write_data_to_csv(timestamp, sample):
    with open('eeg_data1.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp] + list(sample))

try:
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        lsl_start_time = time.time()
        sample, timestamp = inlet.pull_sample()
        print(timestamp, sample)
        write_data_to_csv(timestamp, sample)
        lsl_end_time = time.time()
        lsl_duration = lsl_end_time - lsl_start_time
        print(f"LSL method duration: {lsl_duration:.4f} seconds")
except KeyboardInterrupt as e:
    print("Ending program")
    raise e