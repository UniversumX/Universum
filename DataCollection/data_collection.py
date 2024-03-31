
from neurosity import NeurositySDK
from dotenv import load_dotenv
import os
from neurosity_sdk import NeurosityCrown  # This is hypothetical; replace with the actual import
from influxdb_client import InfluxDBClient
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import time
import argparse
import signal
import sys

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

NEUROSITY_EMAIL=your email here
NEUROSITY_PASSWORD=your password here
NEUROSITY_DEVICE_ID=your device id here


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


# Initialize and connect to Neurosity Crown
crown = NeurosityCrown(device_id="your_device_id", auth="your_auth")
crown.connect()


def write_data_to_influx(label, data):
    point = Point(label)
    for key, value in data.items():
        point = point.field(key, value)
    write_api.write(bucket=bucket, org=org, record=point)

def handle_eeg_data(data):
    # Example processing, adapt according to actual data format
    eeg_data = {"value": sum(data['data'][0])/len(data['data'][0])}  # Simplified example
    write_data_to_influx("EEG", eeg_data)

def handle_accelerometer_data(data):
    # Example processing, adapt according to actual data format
    accel_data = {"x": data['data']['x'], "y": data['data']['y'], "z": data['data']['z']}
    write_data_to_influx("Accelerometer", accel_data)

def signal_handler(sig, frame):
    print('Emergency stop detected. Cleaning up...')
    client.close()
    print('Cleanup done. Exiting.')
    exit(0)


def main(collection_duration):
    unsubscribe_eeg = neurosity.brainwaves_raw(handle_eeg_data)
    unsubscribe_accelerometer = neurosity.accelerometer(handle_accelerometer_data)

    # Run for the specified duration
    time.sleep(collection_duration)

    # Cleanup
    unsubscribe_eeg()
    unsubscribe_accelerometer()
    client.close()
    print("Data collection completed.")




if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Collect EEG and accelerometer data")
    parser.add_argument("--duration", type=int, default=60, help="Duration to collect data for (in seconds)")
    args = parser.parse_args()

    neurosity.login({
        "email": os.getenv("NEUROSITY_EMAIL"),
        "password": os.getenv("NEUROSITY_PASSWORD")
    }).then(lambda _: main(args.duration))