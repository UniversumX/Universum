import os
import signal
import sys
import asyncio
from datetime import datetime
import csv
from pathlib import Path

def write_data_to_csv(data_type, data):
    filename = f"{data_type}_data.csv"
    file_exists = Path(filename).exists()


    if data_type == 'EEG':
    ### TODO: Write EEG data to a CSV file ###
        with open(filename, mode='a', newline='') as file:
            fieldnames = ['timestamp', 'device_id', 'data_type', 'channel', 'value']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()    
        writer.writerow(data)

    elif data_type == 'Accelerometer':
        with open(filename, mode='a', newline='') as file:
            fieldnames = ['timestamp', 'device_id', 'x', 'y', 'z']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()    
        writer.writerow(data)

    
