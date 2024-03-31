
from neurosity import NeurositySDK
from dotenv import load_dotenv
import os
from neurosity_sdk import NeurosityCrown  # This is hypothetical; replace with the actual import
from influxdb_client import InfluxDBClient
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client import WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import os
from neurosity_sdk import NeurosityCrown  # This is hypothetical; replace with the actual import
from influxdb_client import InfluxDBClient
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client import WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import time
import argparse
import signal
import sys
import asyncio
from datetime import datetime



"""
Ensure you have the necessary packages installed in your environment:

influxdb-client for InfluxDB interaction.
python-dotenv for environment variable management.
"""

# Assuming you have set up InfluxDB credentials as environment variables
influxdb_url = os.getenv('INFLUXDB_URL')
token = os.getenv('INFLUXDB_TOKEN')
org = os.getenv('INFLUXDB_ORG')
bucket = os.getenv('INFLUXDB_BUCKET')

# Setting up argparse to accept a collection duration from the command line
parser = argparse.ArgumentParser(description='EEG and Accelerometer Data Collection')
parser.add_argument('--duration', type=int, help='Duration for data collection in seconds', default=60)
args = parser.parse_args()

# Use args.duration as the time to run the data collection
collection_duration = args.duration

NEUROSITY_EMAIL=#your email here
NEUROSITY_PASSWORD=#your password here
NEUROSITY_DEVICE_ID=#your device id here


load_dotenv()

neurosity = NeurositySDK({
    "device_id": os.getenv("NEUROSITY_DEVICE_ID")
})

neurosity.login({
    "email": os.getenv("NEUROSITY_EMAIL"),
    "password": os.getenv("NEUROSITY_PASSWORD")
})


# Initialize InfluxDB client
client = InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)


# Initialize InfluxDB client
client = InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)


# Initialize and connect to Neurosity Crown
info = neurosity.get_info()
print(info)



def write_data_to_influx(label, data):
    point = Point(label)
    for key, value in data.items():
        point = point.field(key, value)
    write_api.write(bucket=bucket, org=org, record=point)

def handle_eeg_data(data):
    # Assuming 'data' contains EEG samples and 'info' contains metadata
    timestamp = datetime.utcnow()
    for sample_set in data['data']:
        for channel_index, value in enumerate(sample_set):
            channel_name = data['info']['channelNames'][channel_index]
            point = (
                Point("EEG")
                .tag("device", os.getenv("NEUROSITY_DEVICE_ID"))
                .tag("channel", channel_name)
                .field("value", float(value))
                .time(timestamp, WritePrecision.NS)
            )
            write_api.write(bucket=bucket, org=org, record=point)

def handle_accelerometer_data(data):
    # Directly uses 'data' assuming it contains 'x', 'y', 'z' acceleration values
    timestamp = datetime.utcnow()
    point = (
        Point("Accelerometer")
        .tag("device", os.getenv("NEUROSITY_DEVICE_ID"))
        .field("x", float(data['x']))
        .field("y", float(data['y']))
        .field("z", float(data['z']))
        .time(timestamp, WritePrecision.NS)
    )
    write_api.write(bucket=bucket, org=org, record=point)

def signal_handler(sig, frame):
    print('Emergency stop detected. Cleaning up...')
    client.close()
    print('Cleanup done. Exiting.')
    exit(0)




async def main(duration):
    await neurosity.login({
        "email": os.getenv("NEUROSITY_EMAIL"),
        "password": os.getenv("NEUROSITY_PASSWORD"),
    })
    
    # Subscribe to EEG and accelerometer data
    unsubscribe_eeg = await neurosity.brainwaves_raw(handle_eeg_data)
    unsubscribe_accel = await neurosity.accelerometer(handle_accelerometer_data)

    # Wait for the specified duration
    await asyncio.sleep(duration)

    # Cleanup
    unsubscribe_eeg()
    unsubscribe_accel()
    write_api.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect EEG and Accelerometer data.")
    parser.add_argument("--duration", type=int, default=60, help="Duration to collect data in seconds.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    asyncio.run(main(args.duration))

    # might get rid of this
    neurosity.login({
        "email": os.getenv("NEUROSITY_EMAIL"),
        "password": os.getenv("NEUROSITY_PASSWORD")
    }).then(lambda _: main(args.duration))


    """
    Notes:
Replace handle_eeg_data and handle_accelerometer_data with the correct logic for handling your data. The given examples are placeholders and need to be adapted to the actual data structure provided by the Neurosity SDK.
The write_data_to_influx function is a simplified method to write data to InfluxDB. Ensure the data structure (data dictionary passed to it) matches what you intend to store.
This script uses a blocking time.sleep for simplicity. For a more responsive or complex application, consider using asynchronous programming patterns or threading.
Running this script directly from the CLI and providing the --duration argument lets you specify how long you want to collect data for.
Pressing CTRL+C will trigger the emergency stop, immediately terminating the data collection and ensuring a clean exit.
    """

    
    """
        Notes:
Replace handle_eeg_data and handle_accelerometer_data with the correct logic for handling your data. The given examples are placeholders and need to be adapted to the actual data structure provided by the Neurosity SDK.
The write_data_to_influx function is a simplified method to write data to InfluxDB. Ensure the data structure (data dictionary passed to it) matches what you intend to store.
This script uses a blocking time.sleep for simplicity. For a more responsive or complex application, consider using asynchronous programming patterns or threading.
Running this script directly from the CLI and providing the --duration argument lets you specify how long you want to collect data for.
Pressing CTRL+C will trigger the emergency stop, immediately terminating the data collection and ensuring a clean exit.

    """