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
# from influx_data import *
from local_storage import *






"""
Ensure you have the necessary packages installed in your environment:

influxdb-client for InfluxDB interaction.
python-dotenv for environment variable management.
"""


# Setting up argparse to accept a collection duration from the command line
parser = argparse.ArgumentParser(description='EEG and Accelerometer Data Collection')
parser.add_argument('--duration', type=int, help='Duration for data collection in seconds', default=60)
args = parser.parse_args()

# Use args.duration as the time to run the data collection
collection_duration = args.duration


# Load environment variables from .env file
load_dotenv()

# Access the variables
neurosity_email = os.getenv('NEUROSITY_EMAIL')
neurosity_password = os.getenv('NEUROSITY_PASSWORD')
neurosity_device_id = os.getenv('NEUROSITY_DEVICE_ID')



neurosity = NeurositySDK({
    "device_id": neurosity_device_id
})


neurosity.login({
    "email": neurosity_email,
    "password": neurosity_password
})

# Show neurosit info on demand
def info_neurosity():
    info = neurosity.get_info()
    print(info)

def handle_eeg_data(data):
    timestamp = datetime.utcnow()
    channel_names = data['info']['channelNames']
    pairs_per_channel = len(data['data'][0]) // len(channel_names)


    for sample_set in data['data']:
        for i in range(len(channel_names)):
            values = sample_set[i * pairs_per_channel:(i + 1) * pairs_per_channel]
            channel_name = channel_names[i]

            # Handling each value in values, you may need to adjust based on your actual requirements:
            for value in values:
                # Write to CSV
                write_data_to_csv('EEG', {
                    'timestamp': NeurositySDK.get_server_timestamp(neurosity),
                    'device_id': neurosity_device_id,
                    'data_type': 'EEG',
                    'channel': channel_name,
                    'value': value,  # Here we use each individual value
                    'x': '',
                    'y': '',
                    'z': ''
                })
                print(f"{timestamp}: Channel {channel_name} - Value {value}")

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

# def handle_eeg_data(data):
#     print(f"Received data sample size: {len(data['data'][0])} values")
#     timestamp = datetime.utcnow()
#     channel_names = data['info']['channelNames']
#     for sample_set in data['data']:
#         for channel_index, value in enumerate(sample_set):
#             if channel_index < len(channel_names):  # Ensure index is valid
#                 channel_name = channel_names[channel_index]
#                 # point = (
#                 #     Point("EEG")
#                 #     .tag("device", os.getenv("NEUROSITY_DEVICE_ID"))
#                 #     .tag("channel", channel_name)
#                 #     .field("value", float(value))
#                 #     .time(timestamp, WritePrecision.NS)
#                 # )
#                 # Write to CSV
#                 write_data_to_csv('EEG', {
#                     'timestamp': timestamp,
#                     'device_id': NEUROSITY_DEVICE_ID,
#                     'data_type': 'EEG',
#                     'channel': channel_name,
#                     'value': value,
#                     'x': '',
#                     'y': '',
#                     'z': ''
#                 })
#             else:
#                 print(f"Skipping index {channel_index}, out of range for channel names.")

# def handle_eeg_data(data):
#     # Assuming 'data' contains EEG samples and 'info' contains metadata
#     timestamp = datetime.utcnow()
#     for sample_set in data['data']:
#         for channel_index, value in enumerate(sample_set):
#             print(len(data), type(data))
#             channel_name = data['info']['channelNames'][channel_index]
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

#             # write to influx
#            #  write_api.write(bucket=bucket, org=org, record=point)
            

def handle_accelerometer_data(data):
    # Directly uses 'data' assuming it contains 'x', 'y', 'z' acceleration values
    timestamp = datetime.utcnow()
    # point = (
    #     Point("Accelerometer")
    #     .tag("device", os.getenv("NEUROSITY_DEVICE_ID"))
    #     .field("x", float(data['x']))
    #     .field("y", float(data['y']))
    #     .field("z", float(data['z']))
    #     .time(timestamp, WritePrecision.NS)
    # )
    # Write to CSV
    write_data_to_csv('Accelerometer', {
        'timestamp': NeurositySDK.get_server_timestamp(neurosity),
        'device_id': neurosity_device_id,
        'data_type': 'Accelerometer',
        'channel': '',
        'value': '',
        'x': data['x'],
        'y': data['y'],
        'z': data['z']
    })

    # write to influx
    # write_api.write(bucket=bucket, org=org, record=point)
    

def signal_handler(sig, frame):
    print('Emergency stop detected. Cleaning up...')
    # client.close()
    print('Cleanup done. Exiting.')
    exit(0)




async def eeg(duration):
    # await neurosity.login({
    #     "email": os.getenv("NEUROSITY_EMAIL"),
    #     "password": os.getenv("NEUROSITY_PASSWORD"),
    # })
    
    # Subscribe to EEG and accelerometer data
    unsubscribe_eeg = neurosity.brainwaves_raw(handle_eeg_data)
    unsubscribe_accel = neurosity.accelerometer(handle_accelerometer_data)

    ### FIX: This does not work as expected. ###
    # Wait for the specified duration
    await asyncio.sleep(duration)

    # Cleanup
    unsubscribe_eeg()
    unsubscribe_accel()
    # write_api.close()
    
def collect(duration):
    asyncio.run(eeg(duration))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect EEG and Accelerometer data.")
    parser.add_argument("--duration", type=int, default=60, help="Duration to collect data in seconds.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    asyncio.run(collect(args.duration))


    """
    Notes:
Replace handle_eeg_data and handle_accelerometer_data with the correct logic for handling your data. The given examples are placeholders and need to be adapted to the actual data structure provided by the Neurosity SDK.
The write_data_to_influx function is a simplified method to write data to InfluxDB. Ensure the data structure (data dictionary passed to it) matches what you intend to store.
This script uses a blocking time.sleep for simplicity. For a more responsive or complex application, consider using asynchronous programming patterns or threading.
Running this script directly from the CLI and providing the --duration argument lets you specify how long you want to collect data for.
Pressing CTRL+C will trigger the emergency stop, immediately terminating the data collection and ensuring a clean exit.
    """
