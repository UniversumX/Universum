
from neurosity import NeurositySDK
from dotenv import load_dotenv
import os
from neurosity_sdk import NeurosityCrown  # This is hypothetical; replace with the actual import
from influxdb_client import InfluxDBClient
import time

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

# Configure your database connection
db_client = InfluxDBClient(url="your_db_url", token="your_db_token", org="your_org")

# Initialize and connect to Neurosity Crown
crown = NeurosityCrown(device_id="your_device_id", auth="your_auth")
crown.connect()


def callback(data):
    print("data", data)

def collect_and_store_data():
    while True:
        eeg_data = crown.get_eeg_data()  # Hypothetical method
        accel_data = crown.get_accelerometer_data()  # Hypothetical method
        timestamp = time.time()  # Ensure synchronization with device time if necessary

        # Create database points (InfluxDB example)
        eeg_point = {
            "measurement": "EEG",
            "time": timestamp,
            "fields": eeg_data
        }
        accel_point = {
            "measurement": "Accelerometer",
            "time": timestamp,
            "fields": accel_data
        }

        # Write points to the database
        db_client.write_points([eeg_point, accel_point])

# Run the data collection and storage loop
collect_and_store_data()