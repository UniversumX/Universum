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





"""
Ensure you have the necessary packages installed in your environment:

influxdb-client for InfluxDB interaction.
python-dotenv for environment variable management.
"""


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

sub = subject.Subject()
datawriter = local_storage.DataWriter(sub) 

def experiment_setup(subject_id = '0000', visit = 0, age = 0, trial = 0):
    # Initialize the subject
    sub.set_subject_id(subject_id)
    sub.set_visit(visit)
    sub.set_age(age)
    # Initialize the data writer
    datawriter.set_subject(sub)
    # Initialize the trial number
    datawriter.set_trial(trial)


# Show neurosit info on demand
def info_neurosity():
    info = neurosity.get_info()
    print(info)
    
def handle_eeg_data(data):
    print("data", data)
    # start = time.time()
    timestamp = datetime.fromtimestamp(data['info']['startTime'] / 1000.0).strftime('%F %T.%f')[:-3]
    channel_names = data['info']['channelNames']
    label = data['label']
    data_by_channel = data['data']
    sample_number = len(data_by_channel[0])

    for i in range(sample_number):
        row = dict()
        row['timestamp'] = timestamp
        for j in range(len(channel_names)):
            row[channel_names[j]] = data_by_channel[j][i]
    # Handling each value in values, you may need to adjust based on your actual requirements:
        datawriter.write_data_to_csv(data_type = 'EEG', data = row, label = label)

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
    # print("data", data)
    timestamp = datetime.fromtimestamp(data['timestamp'] / 1000.0).strftime('%F %T.%f')[:-3]
    # point = (
    #     Point("Accelerometer")
    #     .tag("device", os.getenv("NEUROSITY_DEVICE_ID"))
    #     .field("x", float(data['x']))
    #     .field("y", float(data['y']))
    #     .field("z", float(data['z']))
    #     .time(timestamp, WritePrecision.NS)
    # )
    # Write to CSV
    datawriter.write_data_to_csv('Accelerometer', {
        'timestamp': timestamp,
        'device_id': neurosity_device_id,
        'x': data['x'],
        'y': data['y'],
        'z': data['z'],
        'pitch': data['pitch'],
        'roll': data['roll'],
        'acceleration': data['acceleration'],
        'inclination': data['inclination'], 
    })

    # write to influx
    # write_api.write(bucket=bucket, org=org, record=point)
    

def signal_handler(sig, frame):
    print('Emergency stop detected. Cleaning up...')
    # client.close()
    neurosity_stop()
    print('Cleanup done. Exiting.')
    exit(0)

def neurosity_stop():
    print('Emergency stop detected. Cleaning up...')
    neurosity.remove_all_subscriptions()
    print('Cleanup done.')


async def eeg(duration):
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

def discard_last_trial():
    datawriter.discard_last_trial()

    ### TODO: Implement a way to cancel asyncio tasks ###
def shutdown(loop):
    print('received stop signal, cancelling tasks...')
    for task in asyncio.all_tasks():
        task.cancel()
    print('bye, exiting in a minute...')


async def collect(duration):
    callback = eeg(duration)
    datawriter.set_trial(datawriter.get_trial() + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect EEG and Accelerometer data.")
    parser.add_argument("--duration", type=int, default=60, help="Duration to collect data in seconds.")
    parser.add_argument("--subject_id", type=str, default="0000", help="Subject ID")
    parser.add_argument("--visit", type=int, default=0, help="Visit number")
    parser.add_argument("--trial", type=int, default=0, help="Trial number")
    parser.add_argument("--age", type=int, default=0, help="Age of the subject")
    args = parser.parse_args()

    experiment_setup(args.subject_id, args.visit, args.age, args.trial)

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
