import tkinter as tk
import time
from data_collection import *

class TimerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Timer App")
        
        self.default_time = 60
        self.time_remaining = self.default_time
        self.is_running = False
        
        self.timer_label = tk.Label(self.root, text="Time Remaining: 60")
        self.timer_label.pack()
        
        self.start_button = tk.Button(self.root, text="Start", command=self.start_timer)
        self.start_button.pack()

        self.info_button = tk.Button(self.root, text="Info", command=info_neurosity)
        self.info_button.pack()       

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_timer)
        self.stop_button.pack()
        
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_timer)
        self.reset_button.pack()
        
        self.default_time_label = tk.Label(self.root, text="Default Block Time:")
        self.default_time_label.pack()

        self.default_time_entry = tk.Entry(self.root)
        self.default_time_entry.pack()

        self.submit_button = tk.Button(self.root, text="Submit", command=self.set_default_time)
        self.submit_button.pack()
        
    def set_default_time(self):
        new_time = int(self.default_time_entry.get())
        self.default_time = new_time
        self.time_remaining = new_time
        self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
        self.default_time_entry.delete(0, tk.END)


        
    def start_timer(self):
        collect(self.default_time)
        if not self.is_running:
            self.is_running = True
            self.root.after(1000, self.update_timer)
            
    def stop_timer(self):
        self.is_running = False
        
    def reset_timer(self):
        self.is_running = False
        self.time_remaining = self.default_time
        self.update_timer()
        
    def update_timer(self):
        if self.is_running:
            self.time_remaining -= 1
            self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
            self.root.after(1000, self.update_timer)
            if self.time_remaining == 0:
                self.is_running = False
        else:
            self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
        
root = tk.Tk()
app = TimerApp(root)
root.mainloop()
