import sys
import os
import pandas as pd
from datetime import datetime
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QMessageBox
from data_collection import *
from app import push_action_data

def get_formatted_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class TimerApp(QWidget):
    def __init__(self, data_path, default_time=60, id=None, visit=None, trial=None):
        super().__init__()
        
        self.setWindowTitle("Timer App")
        
        self.default_time = default_time
        self.time_remaining = self.default_time
        self.is_running = False
        
        # Set up layout
        self.layout = QVBoxLayout()
        
        self.timer_label = QLabel(f"Time Remaining: {self.default_time}")
        self.layout.addWidget(self.timer_label)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_timer)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_timer)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_timer)
        self.reset_button.setEnabled(False)
        self.layout.addWidget(self.reset_button)

        self.info_button = QPushButton("Info")
        self.info_button.clicked.connect(info_neurosity)
        self.layout.addWidget(self.info_button)

        self.discard_button = QPushButton("Discard Last Trial")
        self.discard_button.clicked.connect(self.discard_last_trial)
        self.discard_button.setEnabled(False)
        self.layout.addWidget(self.discard_button)

        self.textbox = QTextEdit()
        self.layout.addWidget(self.textbox)
        
        self.setLayout(self.layout)

        # Initialize data collection
        self.action_data = pd.DataFrame(columns=["timestamp", "action_value"])
        self.current_action_value = -1  # no action
        self.procedure_index = 0

        self.data_path = data_path
        self.id = id
        self.visit = visit
        self.trial = trial
        
        self.threads = []

    def collect(self):
        datawriter.check_directory()
        self.unsubscribe_brainwaves = neurosity.brainwaves_raw(handle_eeg_data)
        self.unsubscribe_accelerometer = neurosity.accelerometer(handle_accelerometer_data)

    def stop(self):
        self.unsubscribe_brainwaves()
        self.unsubscribe_accelerometer()
        self.save_action_data()

    def start_timer(self):
        if self.time_remaining == 0:
            self.time_remaining = self.default_time
            self.timer_label.setText(f"Time Remaining: {self.default_time}")
        
        if not self.is_running:
            self.collect()
            self.is_running = True
            self.start_button.setEnabled(False)
            self.info_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.discard_button.setEnabled(False)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_timer)
            self.timer.start(1000)

    def stop_timer(self):
        self.stop()
        self.is_running = False
        self.start_button.setEnabled(True)
        self.info_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.discard_button.setEnabled(True)
        self.save_action_data()

    def save_action_data(self):
        self.action_data.to_csv(os.path.join(self.data_path, "action_data.csv"), index=False)

        for _, row in self.action_data.iterrows():
            data = (
                row['timestamp'],
                row['action_value'],
                self.id,
                self.visit,
                self.trial
            )
            push_action_data(data)

    def reset_timer(self):
        if self.time_remaining != 0:
            discard_last_trial()
        self.is_running = False
        self.time_remaining = self.default_time
        self.start_button.setEnabled(True)
        self.info_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.discard_button.setEnabled(False)
        self.update_timer()
        self.procedure_index = 0
        self.textbox.clear()

    def discard_last_trial(self):
        discard_last_trial()
        self.discard_button.setEnabled(False)

    def update_timer(self):
        if self.is_running:
            self.time_remaining -= 1
            self.timer_label.setText(f"Time Remaining: {self.time_remaining}")
            if self.time_remaining == 0:
                self.stop()
                trial_progress()
                self.is_running = False
                self.start_button.setEnabled(True)
                self.info_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.reset_button.setEnabled(True)
                self.discard_button.setEnabled(False)

            timestamp = self.default_time - self.time_remaining
            if self.procedure_index <= len(procedures) and timestamp >= procedures[self.procedure_index][0]:
                self.update_animation(actions, procedures)
                self.current_action_value = actions[procedures[self.procedure_index][1]].action_value
                self.procedure_index += 1

            new_row = pd.DataFrame(
                [
                    {
                        "timestamp": get_formatted_timestamp(),
                        "action_value": self.current_action_value,
                    }
                ]
            )
            self.action_data = pd.concat([self.action_data, new_row], ignore_index=True)

    def update_animation(self, actions, procedures):
        action = actions[procedures[self.procedure_index][1]]
        if action.text != None:
            self.textbox.clear()
            self.textbox.setText(action.text)
        elif action.audio != None:
            print("play audio at path: " + action.audio)
        elif action.image != None:
            print("display image at path: " + action.image)
        else:
            print("Invalid action")


class InfoApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Info App")

        # Layout setup
        self.layout = QVBoxLayout()

        self.subject_id_label = QLabel("Subject ID:")
        self.layout.addWidget(self.subject_id_label)

        self.id_entry = QLineEdit()
        self.layout.addWidget(self.id_entry)

        self.visit_label = QLabel("Visit Number:")
        self.layout.addWidget(self.visit_label)

        self.visit_entry = QLineEdit()
        self.layout.addWidget(self.visit_entry)

        self.trial_label = QLabel("Trial Number:")
        self.layout.addWidget(self.trial_label)

        self.trial_entry = QLineEdit()
        self.layout.addWidget(self.trial_entry)

        self.default_time_label = QLabel("Default Block Time:")
        self.layout.addWidget(self.default_time_label)

        self.default_time_entry = QLineEdit()
        self.layout.addWidget(self.default_time_entry)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.validate_submit)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def new_window(self):
        self.close()  # Close current window
        self.timer_app = TimerApp(data_path=os.path.join("data", self.id, self.visit, self.trial), 
                                  default_time=int(self.default_time), id=self.id, visit=self.visit, trial=self.trial)
        self.timer_app.show()

    def validate_submit(self):
        self.id = self.id_entry.text()
        self.visit = self.visit_entry.text()
        self.trial = self.trial_entry.text()
        self.default_time = self.default_time_entry.text()

        if self.default_time == "":
            self.default_time = "60"
        if self.id == "":
            self.id = "0000"
        if self.visit == "":
            self.visit = "1"
        if self.trial == "":
            self.trial = "0"

        if self.id.isdigit() and self.visit.isdigit() and self.trial.isdigit() and self.default_time.isdigit():
            experiment_setup(self.id, int(self.visit), int(self.trial))
            QMessageBox.information(self, "Info Entry Successful", "Welcome to the Experiment!\nClick OK to continue")
            self.new_window()
        else:
            QMessageBox.warning(self, "Info Failed", "Invalid Data Format. Please try again with only integers.")


def main():
    app = QApplication(sys.argv)
    info_app = InfoApp()
    info_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
