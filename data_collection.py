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

# from Modules import influx_data
from modules import local_storage, subject
import threading
from pylsl import StreamInlet, resolve_stream

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


# Show neurosity info on demand
def info_neurosity():
    info = neurosity.get_info()
    print(info)


def handle_eeg_data(data):
    # print("data", data)
    # start = time.time()

    # print(timestamp, sample)
    # Modified Code based on LSL realtime collection

    try:
        # first resolve an EEG stream on the lab network
        streams = resolve_stream('type', 'EEG')
        # create a new inlet to read from the stream
        inlet = StreamInlet(streams[0])


        while True:
            sample, timestamp = inlet.pull_sample()
            row = {
                "timestamp": datetime.fromtimestamp(timestamp).strftime("%F %T.%f")[:-3],
            }


            channel_names = streams[0].desc().child("channels").find_children("channel")
            for i, channel in enumerate(channel_names):
                channel_name = channel.get_child_value("label")
                row[channel_name] = sample[i]
            # Write the data row to CSV
            datawriter.write_data_to_csv(data_type="EEG", data=row, label="EEG")
    except Exception as e:
        print(f"An error occurred: {e}")
"""
        for i in range(sample_number):
            row = dict()
            row["timestamp"] = timestamp
            for j in range(len(channel_names)):
                row[channel_names[j]] = data_by_channel[j][i]
            # Handling each value in values, you may need to adjust based on your actual requirements:
            datawriter.write_data_to_csv(data_type="EEG", data=row, label=label)
"""
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
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    sample, timestamp = inlet.pull_sample()

    datawriter.check_directory()
    # Subscribe to EEG and accelerometer data
    unsubscribe_brainwaves = neurosity.brainwaves_raw(handle_eeg_data)
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