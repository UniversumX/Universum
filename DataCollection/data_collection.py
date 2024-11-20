from neurosity import NeurositySDK
from dotenv import load_dotenv
import os
import time
import argparse
import signal
import sys
import asyncio
from datetime import datetime
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from app import push_accelerometer_data, push_eeg_data

# from Modules import influx_data
from modules import local_storage, subject
import threading

from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import csv
import time

"""
Ensure you have the necessary packages installed in your environment:

influxdb-client for InfluxDB interaction.
python-dotenv for environment variable management.
"""


# Load environment variables from .env file
load_dotenv()

# Access the variables
neurosity_email = os.getenv("NEUROSITY_EMAIL")
neurosity_password = os.getenv("NEUROSITY_PASSWORD")
neurosity_device_id = os.getenv("NEUROSITY_DEVICE_ID")


neurosity = NeurositySDK({"device_id": neurosity_device_id})


neurosity.login({"email": neurosity_email, "password": neurosity_password})

sub = subject.Subject()
datawriter = local_storage.DataWriter(sub)

# Initialize live visualization aspects
plt.ion()
fig, axs = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
channels = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
channel_data = {channel: [] for channel in channels}


def experiment_setup(subject_id="0000", visit=1, trial=1):
    # Initialize the subject
    sub.set_subject_id(subject_id)
    sub.set_visit(visit)
    # Initialize the data writer
    datawriter.set_subject(sub)
    # Initialize the trial number
    datawriter.set_trial(trial)
    # Record subject info
    datawriter.write_subject_info()


# Show neurosit info on demand
def info_neurosity():
    info = neurosity.get_info()
    print(info)


def live_plot_eeg_data(row):
    for channel in channels:
        scaled_value = row[channel] / 100000.0
        channel_data[channel].append(scaled_value)

    for ax, channel in zip(axs, channels):
        ax.cla()
        ax.plot(channel_data[channel], "r")
        ax.set_title(channel)
        ax.set_ylim(-1, 1)

    plt.pause(0.1)


def write_data_to_csv(timestamp, sample, channel_names=None, label=None):
    with open('eeg_data1.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if channel_names and label:
            # If channel names and label are available, write header with these as columns
            writer.writerow([timestamp] + [label] + list(sample))  # Optional: include the label
        else:
            # If channel names are not available, just write the sample data with timestamp
            writer.writerow([timestamp] + list(sample))


def collect_lsl_data(data):
    try:
        # Resolve the EEG stream
        print("Looking for EEG stream...")
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])

        while True:
            # Pull new sample
            lsl_start_time = time.time()
            sample, timestamp = inlet.pull_sample()
            print("Timestamp:", timestamp, "Sample:", sample)

            # Call write_data_to_csv to write the sample to CSV
            write_data_to_csv(timestamp, sample)

            lsl_end_time = time.time()
            lsl_duration = lsl_end_time - lsl_start_time
            print(f"LSL method duration: {lsl_duration:.4f} seconds")

    except KeyboardInterrupt as e:
        print("Ending program")
        raise e

def handle_eeg_data(data):
    # print("data", data)
    # start = time.time()

    timestamp = datetime.fromtimestamp(data["info"]["startTime"] / 1000.0).strftime("%F %T.%f")[:-3]
    channel_names = data["info"]["channelNames"]
    label = data["label"]
    data_by_channel = data["data"]
    sample_number = len(data_by_channel[0])
    print(sample_number)

    for i in range(sample_number):
        row = dict()
        row["timestamp"] = timestamp
        for j in range(len(channel_names)):
            row[channel_names[j]] = data_by_channel[j][i]

        live_plot_eeg_data(row)
        # Handling each value in values, you may need to adjust based on your actual requirements:
        datawriter.write_data_to_csv(data_type="EEG", data=row, label=label)

    # end = time.time()
    # print(end - start)


# def handle_eeg_data(data):
#     timestamp = datetime.utcnow()
#     channel_names = data['info']['channelNames']
#     pairs_per_channel = len(data['data'][0]) // len(channel_names)

#     for sample_set in data['data']:
#         for i in range(len(channel_names)):
#             values = sample_set[i*pairs_per_channel:(i+1)*pairs_per_channel]
#             channel_name = channel_names[i]
#             # point = (
#             #     Point("EEG")
#             #     .tag("device", os.getenv("NEUROSITY_DEVICE_ID"))
#             #     .tag("channel", channel_name)
#             #     .field("value", float(value))
#             #     .time(timestamp, WritePrecision.NS)
#             # )
#             # Write to CSV
#             write_data_to_csv('EEG', {
#                 'timestamp': timestamp,
#                 'device_id': NEUROSITY_DEVICE_ID,
#                 'data_type': 'EEG',
#                 'channel': channel_name,
#                 'value': value,
#                 'x': '',
#                 'y': '',
#                 'z': ''
#             })
#             print(f"{timestamp}: Channel {channel_name} - Values {values}")

#             # write to influx
#            #  write_api.write(bucket=bucket, org=org, record=point)


def handle_accelerometer_data(data):
    # Directly uses 'data' assuming it contains 'x', 'y', 'z' acceleration values
    print("data", data)
    timestamp = datetime.fromtimestamp(data["timestamp"] / 1000.0).strftime("%F %T.%f")[
        :-3
    ]
    # point = (
    #     Point("Accelerometer")
    #     .tag("device", os.getenv("NEUROSITY_DEVICE_ID"))
    #     .field("x", float(data['x']))
    #     .field("y", float(data['y']))
    #     .field("z", float(data['z']))
    #     .time(timestamp, WritePrecision.NS)
    # )
    # Write to CSV
    datawriter.write_data_to_csv(
        "Accelerometer",
        {
            "timestamp": timestamp,
            "device_id": neurosity_device_id,
            "x": data["x"],
            "y": data["y"],
            "z": data["z"],
            "pitch": data["pitch"],
            "roll": data["roll"],
            "acceleration": data["acceleration"],
            "inclination": data["inclination"],
        },
    )

    # write to influx
    # write_api.write(bucket=bucket, org=org, record=point)


def signal_handler(sig, frame):
    print("Emergency stop detected. Cleaning up...")
    # client.close()
    print("Cleanup done. Exiting.")
    exit(0)


def eeg():
    datawriter.check_directory()
    # Subscribe to EEG and accelerometer data
    unsubscribe_brainwaves = neurosity.brainwaves_raw(collect_lsl_data)
    unsubscribe_accelerometer = neurosity.accelerometer(handle_accelerometer_data)
    time.sleep(60)
    # Unsubscribe from the data
    unsubscribe_brainwaves()
    unsubscribe_accelerometer()
    # write_api.close()


def trial_progress():
    datawriter.set_trial(datawriter.get_trial() + 1)


def discard_last_trial():
    datawriter.discard_last_trial()


def collect():
    eeg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect EEG and Accelerometer data.")
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration to collect data in seconds."
    )
    parser.add_argument("--subject_id", type=str, default="0000", help="Subject ID")
    parser.add_argument("--visit", type=int, default=0, help="Visit number")
    parser.add_argument("--trial", type=int, default=0, help="Trial number")
    args = parser.parse_args()

    experiment_setup(args.subject_id, args.visit, args.trial)

    signal.signal(signal.SIGINT, signal_handler)
    collect()

    """
    Notes:
Replace handle_eeg_data and handle_accelerometer_data with the correct logic for handling your data. The given examples are placeholders and need to be adapted to the actual data structure provided by the Neurosity SDK.
The write_data_to_influx function is a simplified method to write data to InfluxDB. Ensure the data structure (data dictionary passed to it) matches what you intend to store.
This script uses a blocking time.sleep for simplicity. For a more responsive or complex application, consider using asynchronous programming patterns or threading.
Running this script directly from the CLI and providing the --duration argument lets you specify how long you want to collect data for.
Pressing CTRL+C will trigger the emergency stop, immediately terminating the data collection and ensuring a clean exit.
    """
