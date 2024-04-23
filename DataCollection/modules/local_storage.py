import os
import csv
from pathlib import Path
import os.path
from modules import subject


class DataWriter:
    _subject = subject.Subject('0000', 0)
    _subdirectory = f"data/{_subject.get_subject_id()}/{_subject.get_visit()}/0/"
    _trial = 0

    def __init__(self, subject, trial = 0):
        self._subject = subject
        self._subdirectory = f"data/{self._subject.get_subject_id()}/{self._subject.get_visit()}/0/"
        self._trial = trial
    
    def set_subject(self, subject):
        self._subject = subject
        self._subdirectory = f"data/{self._subject.get_subject_id()}/{self._subject.get_visit()}/0/"
    
    def set_trial(self, trial):
        self._trial = trial
        self._subdirectory = f"data/{self._subject.get_subject_id()}/{self._subject.get_visit()}/{self._trial}/"

    def get_trial(self):
        return self._trial


    def write_data_to_csv(self, data_type: str, data: dict, label = None): 
            self._subdirectory = f"data/{self._subject.get_subject_id()}/{self._subject.get_visit()}/{self._trial}/"
            if not os.path.exists(self._subdirectory):
                os.makedirs(self._subdirectory) 
            write_data_type = f"write_{data_type.lower()}_data"
            filename = f"{self._subdirectory}/{data_type.lower()}_data.csv"
            if label is not None:
                filename = f"{self._subdirectory}/{data_type.lower()}_data_{label}.csv"
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
                'subject_id': self._subject.get_subject_id(),
                'visit': self._subject.get_visit(),
                'age': self._subject.get_age()
            }
            data_list = [data['subject_id'], data['visit'], data['age']]
            dir = "data/subject_info.csv"
            file_exists = Path(dir).exists()
            with open(dir, mode='a+', newline='') as file:
                fieldnames = ['subject_id', 'visit', 'age'] 
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()    
                csv_reader = csv.reader(file, delimiter=',')
                # convert string to list
                list_of_csv = list(csv_reader)
                if data_list not in list_of_csv:
                    writer.writerow(data)
                

    def discard_last_trial(self):
            path = f"{self._subdirectory}"
            if os.path.exists(path):
                os.remove(path)

