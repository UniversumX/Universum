import tkinter as tk
import time, os
from data_collection import *
from tkinter import messagebox
from datetime import datetime
from app import push_action_data
import pandas as pd
import pygame


def get_formatted_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class TimerApp:
    def __init__(self, root, data_path, default_time=60, id=None, visit=None, trial=None):
        self.root = root
        self.root.title("Timer App")

        self.default_time = default_time
        self.time_remaining = self.default_time
        self.is_running = False

        self.timer_label = tk.Label(
            self.root, text=f"Time Remaining: {self.default_time}"
        )
        self.timer_label.pack()

        self.start_button = tk.Button(self.root, text="Start", command=self.start_timer)
        self.start_button.config(state="normal")
        self.start_button.pack()

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_timer)
        self.stop_button.config(state="disabled")
        self.stop_button.pack()

        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_timer)
        self.reset_button.config(state="disabled")
        self.reset_button.pack()

        self.info_button = tk.Button(self.root, text="Info", command=info_neurosity)
        self.info_button.config(state="normal")
        self.info_button.pack()

        self.discard_button = tk.Button(
            self.root, text="Discard Last Trial", command=self.discard_last_trial
        )
        self.discard_button.config(state="disabled")
        self.discard_button.pack()

        self.textbox = tk.Text(self.root, height=10, width=20)
        self.textbox.pack(pady=10)

        self.action_data = pd.DataFrame(columns=["timestamp", "action_value"])
        self.current_action_value = -1  # no action
        self.procedure_index = 0 # first procedure

        self.data_path = data_path
        self.id = id
        self.visit = visit
        self.trial = trial

        self.threads = []
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def collect(self):
        datawriter.check_directory()
        self.unsubscribe_brainwaves = neurosity.brainwaves_raw(handle_eeg_data)
        self.unsubscribe_accelerometer = neurosity.accelerometer(
            handle_accelerometer_data
        )

    def stop(self):
        self.unsubscribe_brainwaves()
        self.unsubscribe_accelerometer()

        self.save_action_data()

    def start_timer(self):
        if self.time_remaining == 0:
            self.time_remaining = self.default_time
            self.timer_label.config(text=f"Time Remaining: {self.default_time}")
        if not self.is_running:
            self.collect()
            self.is_running = True
            self.start_button.config(state="disabled")
            self.info_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.reset_button.config(state="disabled")
            self.reset_button.config(state="disabled")
            self.root.after(1000, self.update_timer)

    def stop_timer(self):
        for thread in self.threads:
            thread.join()

        self.stop()
        self.is_running = False
        self.start_button.config(state="normal")
        self.info_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.reset_button.config(state="normal")
        self.discard_button.config(state="normal")

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
        self.start_button.config(state="normal")
        self.info_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.reset_button.config(state="disabled")
        self.discard_button.config(state="disabled")
        self.update_timer()

        self.procedure_index = 0
        self.textbox.delete(1.0, tk.END)

    def discard_last_trial(self):
        discard_last_trial()
        self.discard_button.config(state="disabled")

    def update_timer(self):
        if self.is_running:
            self.time_remaining -= 1
            self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
            self.root.after(1000, self.update_timer)
            if self.time_remaining == 0:
                self.stop()
                trial_progress()
                self.is_running = False
                self.start_button.config(state="normal")
                self.info_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self.reset_button.config(state="normal")
                self.discard_button.config(state="disabled")

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
        else:
            self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")

    def update_animation(self, actions, procedures):
        action = actions[procedures[self.procedure_index][1]]
        if action.text != None:
            self.textbox.delete(1.0, tk.END)
            self.textbox.insert(tk.INSERT, action.text)
        elif action.audio != None:
            # play audio
            print("play audio at path: " + action.audio)
            audio_thread = threading.Thread(target=self.play_audio, args=(action.audio,))
            self.threads.append(audio_thread)
            audio_thread.start()
        elif action.image != None:
            # display image
            print("display image at path: " + action.image)
        else:
            print("Invalid action")

    def play_audio(self, audio_path):
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        except pygame.error as e:
            print(f"Error loading audio file {audio_path}: {e}")

    def on_closing(self):
        for thread in self.threads:
            thread.join()
        self.root.destroy()


class InfoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Info App")
        self.subject_id_label = tk.Label(self.root, text="Subject ID:")
        self.subject_id_label.pack()

        self.id_entry = tk.Entry(self.root)
        self.id_entry.pack()

        # Create and place the password label and entry
        self.visit_label = tk.Label(self.root, text="Visit Number:")
        self.visit_label.pack()

        self.visit_entry = tk.Entry(self.root)  # Show asterisks for password
        self.visit_entry.pack()

        self.trial_label = tk.Label(self.root, text="Trial Number:")
        self.trial_label.pack()

        self.trial_entry = tk.Entry(self.root)  # Show asterisks for password
        self.trial_entry.pack()

        self.default_time_label = tk.Label(self.root, text="Default Block Time:")
        self.default_time_label.pack()

        self.default_time_entry = tk.Entry(self.root)
        self.default_time_entry.pack()
        # Create and place the login button
        self.submit_button = tk.Button(
            self.root, text="Submit", command=self.validate_submit
        )
        self.submit_button.pack()

    def new_window(self):
        self.root.destroy()  # close the current window
        self.root = tk.Tk()  # create another Tk instance
        data_path = os.path.join("data", self.id, self.visit, self.trial)
        self.app = TimerApp(self.root, data_path, int(self.default_time), self.id, self.visit, self.trial)  # create Demo2 window
        self.root.mainloop()

    def validate_submit(self):
        self.id = self.id_entry.get()
        self.visit = self.visit_entry.get()
        self.trial = self.trial_entry.get()
        self.default_time = self.default_time_entry.get()

        # You can add your own validation logic here
        if self.default_time == "":
            self.default_time = "60"
        if self.id == "":
            self.id = "0000"
        if self.visit == "":
            self.visit = "1"
        if self.trial == "":
            self.trial = "0"
        if (
            self.id.isdigit()
            and self.visit.isdigit()
            and self.trial.isdigit()
            and self.default_time.isdigit()
        ):
            experiment_setup(self.id, int(self.visit), int(self.trial))
            messagebox.showinfo(
                "Info Entry Successful",
                "Welcome to the Experiment!\nClick OK to continue",
            )
            self.new_window()
        else:
            messagebox.showerror(
                "Info Failed",
                "Invalid Data Format. Please try again with only integers.",
            )


# Create and place the username label and entry


def main():
    pygame.mixer.init()
    root = tk.Tk()
    app = InfoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
