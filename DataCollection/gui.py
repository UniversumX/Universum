import tkinter as tk
import time
from data_collection import *
from tkinter import messagebox

class TimerApp:
    def __init__(self, root, default_time=60):
        self.root = root
        self.root.title("Timer App")
        
        self.default_time = default_time
        self.time_remaining = self.default_time
        self.is_running = False

        self.timer_label = tk.Label(self.root, text=f"Time Remaining: {self.default_time}")
        self.timer_label.pack()
        
        self.start_button = tk.Button(self.root, text="Start", command=self.start_timer)
        self.start_button.config(state = 'normal')
        self.start_button.pack()

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_timer)
        self.stop_button.config(state = 'disabled')
        self.stop_button.pack()
        
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_timer)
        self.reset_button.config(state = 'disabled')
        self.reset_button.pack()

        self.info_button = tk.Button(self.root, text="Info", command=info_neurosity)
        self.info_button.config(state = 'normal')
        self.info_button.pack()       

        self.discard_button = tk.Button(self.root, text="Discard Last Trial", command=self.discard_last_trial)
        self.discard_button.config(state = 'disabled')
        self.discard_button.pack()
        
    def start_timer(self):
        if self.time_remaining == 0:
            self.time_remaining = self.default_time
        if not self.is_running:
            collect()
            self.is_running = True
            self.start_button.config(state = 'disabled')
            self.info_button.config(state = 'disabled')
            self.stop_button.config(state = 'normal')
            self.reset_button.config(state = 'disabled')
            self.reset_button.config(state = 'disabled')
            self.root.after(1000, self.update_timer)
            
    def stop_timer(self):
        ### TODO: Implement a way to stop the asyncio task ###
        neurosity_stop()
        self.is_running = False
        self.start_button.config(state = 'normal')
        self.info_button.config(state = 'normal')
        self.stop_button.config(state = 'disabled')
        self.reset_button.config(state = 'normal')
        self.discard_button.config(state = 'normal')

    def reset_timer(self):
        if self.time_remaining != 0:
            discard_last_trial()
        self.is_running = False
        self.time_remaining = self.default_time
        self.start_button.config(state = 'normal')
        self.info_button.config(state = 'normal')
        self.stop_button.config(state = 'disabled')
        self.reset_button.config(state = 'disabled')
        self.discard_button.config(state = 'disabled')
        self.update_timer()

    def discard_last_trial(self):
        discard_last_trial()
        self.discard_button.config(state = 'disabled')
        
    def update_timer(self):
        if self.is_running:
            self.time_remaining -= 1
            self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
            self.root.after(1000, self.update_timer)
            if self.time_remaining == 0:
                neurosity_stop()
                trial_progress() 
                self.is_running = False
                self.start_button.config(state = 'normal')
                self.info_button.config(state = 'normal')
                self.stop_button.config(state = 'disabled')
                self.reset_button.config(state = 'normal')
                self.discard_button.config(state = 'disabled')
        else:
            self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
    
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

        self.age_label = tk.Label(self.root, text="Age:")
        self.age_label.pack()

        self.age_entry = tk.Entry(self.root)  # Show asterisks for password
        self.age_entry.pack()

        self.trial_label = tk.Label(self.root, text="Trial Number:")
        self.trial_label.pack()

        self.trial_entry = tk.Entry(self.root)  # Show asterisks for password
        self.trial_entry.pack()

        self.default_time_label = tk.Label(self.root, text="Default Block Time:")
        self.default_time_label.pack()

        self.default_time_entry = tk.Entry(self.root)
        self.default_time_entry.pack()
# Create and place the login button
        self.submit_button = tk.Button(self.root, text="Submit", command=self.validate_submit)
        self.submit_button.pack()

    def new_window(self):
        self.root.destroy() # close the current window
        self.root = tk.Tk() # create another Tk instance
        self.app = TimerApp(self.root, int(self.default_time)) # create Demo2 window
        self.root.mainloop()

    def validate_submit(self):
        self.id = self.id_entry.get()
        self.visit = self.visit_entry.get()
        self.age = self.age_entry.get()
        self.trial = self.trial_entry.get()
        self.default_time = self.default_time_entry.get()

# You can add your own validation logic here
        if self.default_time == "":
            self.default_time = '60'
        if self.id == "":
            self.id = '0000'
        if self.visit == "":
            self.visit = '0'
        if self.age == "":
            self.age = '0'
        if self.trial == "":
            self.trial = '0'
        if self.id.isdigit() and self.visit.isdigit() and self.age.isdigit() and self.trial.isdigit() and self.default_time.isdigit():
            experiment_setup(self.id, int(self.visit), int(self.age), int(self.trial))
            messagebox.showinfo("Info Entry Successful", "Welcome to the Experiment!\nClick OK to continue")
            self.new_window()
        else:
            messagebox.showerror("Info Failed", "Invalid Data Format. Please try again with only integers.")
# Create and place the username label and entry
       

def main():
    root = tk.Tk()
    app = InfoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
