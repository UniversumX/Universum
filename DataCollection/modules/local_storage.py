import os
import csv
from pathlib import Path
import os.path
from modules import subject
import shutil
import pandas as pd
import time
from datetime import datetime

class DataWriter:
    _subject = subject.Subject('0000', 1)
    _trial = 0
    _subdirectory = f"data/{_subject.get_subject_id()}/{_subject.get_visit()}/{_trial}/"

    def __init__(self, subject, trial = 0):
        self._subject = subject
        self._trial = trial
        self._subdirectory = f"data/{self._subject.get_subject_id()}/{self._subject.get_visit()}/{self._trial}/"
    
    def set_subject(self, subject):
        self._subject = subject
        self._subdirectory = f"data/{self._subject.get_subject_id()}/{self._subject.get_visit()}/{self._trial}/"
    
    def set_trial(self, trial):
        self._trial = trial
        self._subdirectory = f"data/{self._subject.get_subject_id()}/{self._subject.get_visit()}/{self._trial}/"

    def get_trial(self):
        return self._trial

    def check_directory(self):
        if not os.path.exists(self._subdirectory):
            os.makedirs(self._subdirectory)

    def write_data_to_csv(self, data_type: str, data: dict, label = None): 
            write_data_type = f"write_{data_type.lower()}_data"
            filename = f"{self._subdirectory}{data_type.lower()}_data.csv"
            if label is not None:
                filename = f"{self._subdirectory}{data_type.lower()}_data_{label}.csv"
            file_exists = Path(filename).exists()
            if hasattr(self, write_data_type) and callable(func := getattr(self, write_data_type)):
                func(data, filename, file_exists)

    def write_eeg_data(self, data: dict, filename: str, file_exists: bool): 
            with open(filename, mode='a', newline='') as file:
                fieldnames = ['timestamp', 
                    'CP3', 'C3',
                    'F5',  'PO3',
                    'PO4', 'F6',
                    'C4',  'CP4']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()    
                writer.writerow(data)

    def write_accelerometer_data(self, data: dict, filename: str, file_exists: bool):
            
            with open(filename, mode='a', newline='') as file:
                fieldnames = ['timestamp', 'device_id', 'x', 'y', 'z', 'pitch', 'roll', 'acceleration', 'inclination']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
            
                if not file_exists:
                    writer.writeheader()    
                writer.writerow(data)

    def write_subject_info(self):
            data = {
                'time': datetime.fromtimestamp(time.time()).strftime('%F %T.%f')[:-3],
                'subject_id': self._subject.get_subject_id(),
                'visit': self._subject.get_visit(),
            }
            dir = "data/subject_info.csv"
            file_exists = Path(dir).exists()
            file = open(dir, mode='a', newline='')
            fieldnames = ['time','subject_id', 'visit']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                writer.writerow(data)
            elif not self.check_subject_info(data):
                writer.writerow(data)

    def check_subject_info(self, data):
            df = pd.read_csv("data/subject_info.csv")
            for index, row in df.iterrows():
                if int(row['subject_id']) == int(data['subject_id']) and int(row['visit']) == int(data['visit']):
                    return True
            return False

    def discard_last_trial(self):
            path = f"{self._subdirectory}"
            if os.path.exists(path):
                shutil.rmtree(path)


