import tkinter as tk
from tkinter import ttk # Also personally added
import time
from data_collection import *
from tkinter import messagebox, Label, Canvas
from PIL import Image, ImageTk
import pygame
import time
from actions import Action, actions, procedures

# class TimerApp:
#     def __init__(self, root, default_time=60):
#         self.root = root
#         self.root.title("Timer App")
    
#         self.default_time = default_time
#         self.time_remaining = self.default_time
#         self.is_running = False

#         self.timer_label = tk.Label(self.root, text=f"Time Remaining: {self.default_time}")
#         self.timer_label.pack()
    
#         self.start_button = tk.Button(self.root, text="Start", command=self.start_timer)
#         self.start_button.config(state = 'normal')
#         self.start_button.pack()

#         self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_timer)
#         self.stop_button.config(state = 'disabled')
#         self.stop_button.pack()
    
#         self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_timer)
#         self.reset_button.config(state = 'disabled')
#         self.reset_button.pack()

#         self.info_button = tk.Button(self.root, text="Info", command=info_neurosity)
#         self.info_button.config(state = 'normal')
#         self.info_button.pack()       

#         self.discard_button = tk.Button(self.root, text="Discard Last Trial", command=self.discard_last_trial)
#         self.discard_button.config(state = 'disabled')
#         self.discard_button.pack()
    
#     def collect(self):
#         datawriter.check_directory()
#         self.unsubscribe_brainwaves = neurosity.brainwaves_raw(handle_eeg_data)
#         self.unsubscribe_accelerometer = neurosity.accelerometer(handle_accelerometer_data)

#     def stop(self):
#         self.unsubscribe_brainwaves()
#         self.unsubscribe_accelerometer()

#     def start_timer(self):
#         if self.time_remaining == 0:
#             self.time_remaining = self.default_time
#             self.timer_label.config(text=f"Time Remaining: {self.default_time}")
#         if not self.is_running:  
#             self.collect()
#             self.is_running = True
#             self.start_button.config(state = 'disabled')
#             self.info_button.config(state = 'disabled')
#             self.stop_button.config(state = 'normal')
#             self.reset_button.config(state = 'disabled')
#             self.reset_button.config(state = 'disabled')
#             self.root.after(1000, self.update_timer)
        
#     def stop_timer(self):
#         self.stop()
#         self.is_running = False
#         self.start_button.config(state = 'normal')
#         self.info_button.config(state = 'normal')
#         self.stop_button.config(state = 'disabled')
#         self.reset_button.config(state = 'normal')
#         self.discard_button.config(state = 'normal')

#     def reset_timer(self):
#         if self.time_remaining != 0:
#             discard_last_trial()
#         self.is_running = False
#         self.time_remaining = self.default_time
#         self.start_button.config(state = 'normal')
#         self.info_button.config(state = 'normal')
#         self.stop_button.config(state = 'disabled')
#         self.reset_button.config(state = 'disabled')
#         self.discard_button.config(state = 'disabled')
#         self.update_timer()

#     def discard_last_trial(self):
#         discard_last_trial()
#         self.discard_button.config(state = 'disabled')
    
#     def update_timer(self):
#         if self.is_running:
#             self.time_remaining -= 1
#             self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
#             self.root.after(1000, self.update_timer)
#             if self.time_remaining == 0:
#                 self.stop()
#                 trial_progress() 
#                 self.is_running = False
#                 self.start_button.config(state = 'normal')
#                 self.info_button.config(state = 'normal')
#                 self.stop_button.config(state = 'disabled')
#                 self.reset_button.config(state = 'normal')
#                 self.discard_button.config(state = 'disabled')
#         else:
#             self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")

# class InfoApp:
#     def __init__(self, root): 
#         self.root = root
#         self.root.title("Info App")
#         self.subject_id_label = tk.Label(self.root, text="Subject ID:")
#         self.subject_id_label.pack()

#         self.id_entry = tk.Entry(self.root)
#         self.id_entry.pack()

# # Create and place the password label and entry
#         self.visit_label = tk.Label(self.root, text="Visit Number:")
#         self.visit_label.pack()

#         self.visit_entry = tk.Entry(self.root)  # Show asterisks for password
#         self.visit_entry.pack()

#         self.trial_label = tk.Label(self.root, text="Trial Number:")
#         self.trial_label.pack()

#         self.trial_entry = tk.Entry(self.root)  # Show asterisks for password
#         self.trial_entry.pack()

#         self.default_time_label = tk.Label(self.root, text="Default Block Time:")
#         self.default_time_label.pack()

#         self.default_time_entry = tk.Entry(self.root)
#         self.default_time_entry.pack()
# # Create and place the login button
#         self.submit_button = tk.Button(self.root, text="Submit", command=self.validate_submit)
#         self.submit_button.pack()

#     def new_window(self):
#         self.root.destroy() # close the current window
#         self.root = tk.Tk() # create another Tk instance
#         self.app = TimerApp(self.root, int(self.default_time)) # create Demo2 window
#         self.root.mainloop()

#     def validate_submit(self):
#         self.id = self.id_entry.get()
#         self.visit = self.visit_entry.get()
#         self.trial = self.trial_entry.get()
#         self.default_time = self.default_time_entry.get()

# # You can add your own validation logic here
#         if self.default_time == "":
#             self.default_time = '60'
#         if self.id == "":
#             self.id = '0000'
#         if self.visit == "":
#             self.visit = '1'
#         if self.trial == "":
#             self.trial = '0'
#         if self.id.isdigit() and self.visit.isdigit() and self.trial.isdigit() and self.default_time.isdigit():
#             experiment_setup(self.id, int(self.visit), int(self.trial))
#             messagebox.showinfo("Info Entry Successful", "Welcome to the Experiment!\nClick OK to continue")
#             self.new_window()
#         else:
#             messagebox.showerror("Info Failed", "Invalid Data Format. Please try again with only integers.")
# # Create and place the username label and entry
    
# def main():
#     root = tk.Tk()
#     app = InfoApp(root)
#     root.mainloop()

# if __name__ == "__main__":
#     main()


############################################################
#################### Original Code Above ###################
############################################################


#############################
### Dynamical Trial Pages ###
#############################
        
class TimerApp:
    def __init__(self, root, default_time=60, user_choice=0, trial_num=0):
        self.root = root
        self.root.title(f"Option {user_choice}, Trial: {trial_num}")
        
        self.trial_num = trial_num
        self.default_time = default_time
        self.time_remaining = self.default_time
        self.is_running = False
        self.user_choice = user_choice
        
        self.timer_label = tk.Label(self.root, text=f"Time Remaining: {self.default_time}", font=("Arial", 24))
        self.timer_label.pack(pady=20)
        
        self.start_button = tk.Button(self.root, text="Start", command=self.start_timer)
        self.start_button.pack(pady=5)
        
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_timer, state='disabled')
        self.stop_button.pack(pady=5)
        
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_timer, state='disabled')
        self.reset_button.pack(pady=5)
        
        self.trial_data = self.load_trial_data()
        self.total_trials = len(self.trial_data)
        self.trial_data = self.trial_data[self.trial_num]
        
        
        next_trial_text = "Finished" if (self.trial_num + 1) >= self.total_trials else f"Next Trial: {self.trial_num + 1}"
        self.next_trial_button = tk.Button(self.root, text=next_trial_text, command=self.next_trial)
        self.next_trial_button.pack(pady=5)
        
        self.canvas = Canvas(self.root, width=200, height=200)
        self.canvas.pack(pady=20)
        self.dot = self.canvas.create_oval(90, 90, 110, 110, fill="blue")
        self.canvas.itemconfig(self.dot, state='hidden')
    
    def load_trial_data(self):
        with open("input_data.txt", "r") as file:
            i = 1
            for line in file:
                if(i == self.user_choice):
                    data = line.strip().split(',')
                    return data
                i += 1
        
    def next_trial(self):
        if self.trial_num + 1 < self.total_trials:
            self.root.destroy()
            timer_root = tk.Tk()
            TimerApp(timer_root, user_choice=self.user_choice, trial_num=self.trial_num + 1)
            timer_root.mainloop()
        else:
            self.root.destroy()  # Go back to options screen
            dropdown_root = tk.Tk()  # Create a new Tk instance for OptionsPage
            OptionsPage(dropdown_root)  # Initialize OptionsPage
            dropdown_root.mainloop()
    
    def start_timer(self):
        if self.time_remaining == 0:
            self.time_remaining = self.default_time
            self.timer_label.config(text=f"Time Remaining: {self.default_time}")
        
        if not self.is_running:  
            self.is_running = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.reset_button.config(state='disabled')
            self.update_timer()
    
    def stop_timer(self):
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.reset_button.config(state='normal')
    
    def reset_timer(self):
        self.is_running = False
        self.time_remaining = self.default_time
        self.timer_label.config(text=f"Time Remaining: {self.default_time}")
        self.canvas.itemconfig(self.dot, state='hidden')
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.reset_button.config(state='disabled')
    
    def update_timer(self):
        if self.is_running and self.time_remaining > 0:
            # self.collect()
            self.time_remaining -= 1
            self.timer_label.config(text=f"Time Remaining: {self.time_remaining}")
            self.display_circle()
            
            if self.time_remaining <= 5:
                self.timer_label.config(fg="red")
            else:
                self.timer_label.config(fg="black")
            
            self.root.after(1000, self.update_timer)
        elif self.time_remaining == 0:
            self.timer_label.config(text="Time's up!")
            self.timer_label.config(fg="black")
            self.stop_timer()
            
    # def collect(self):
    #     datawriter.check_directory()
    #     self.unsubscribe_brainwaves = neurosity.brainwaves_raw(handle_eeg_data)
    #     self.unsubscribe_accelerometer = neurosity.accelerometer(handle_accelerometer_data)
    
    def display_circle(self):
        trial_digits = self.trial_data[self.trial_num].strip()
        index = (self.default_time - self.time_remaining) // 3
        
        if index < len(trial_digits):
            digit = trial_digits[index]
            colors = {"0": "", "1": "red", "2": "green", "3": "blue"}
            if digit in colors and digit != "0":
                self.canvas.itemconfig(self.dot, fill=colors[digit], state='normal')
                self.root.after(500, lambda: self.canvas.itemconfig(self.dot, state='hidden'))
            else:
                self.canvas.itemconfig(self.dot, state='hidden')


#####################
### Options Page ###
####################
class OptionsPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Select an Option")
        self.root.geometry("600x400")
        
        total_num_options = 0
        
        # Count total options by reading the file
        with open("input_data.txt") as file:
            for line in file:
                total_num_options += 1
        
        # Create buttons based on available options
        for i in range(3):
            self.root.columnconfigure(i, weight=1, minsize=75)
            self.root.rowconfigure(i, weight=1, minsize=50)
            for j in range(3):
                option_num = j + (i * 3) + 1
                if option_num > total_num_options:
                    break
                
                frame = tk.Frame(self.root)
                frame.grid(row=i, column=j, padx=5, pady=5)
                
                Op_Button = tk.Button(
                    frame,
                    text=f"Option {option_num}",
                    command=lambda option_num=option_num: self.my_open_option_page(option_num),
                    bg="#A0C7C3",
                    activebackground="#7CACAC",
                    font=("Arial", 12, "bold")
                )
                Op_Button.pack(padx=5, pady=5)

    def my_open_option_page(self, option_num=0):
        self.root.destroy()
        op_root = tk.Tk()
        InstructionPage(op_root, option_num)
        op_root.mainloop()

############################################
### Dynamic Instructions Page per Option ###
############################################
class InstructionPage:
    def __init__(self, root, option_num=0):
        self.root = root
        self.root.title(f"Option {option_num} Information Page")
        self.root.geometry("600x400")

        # Create a frame to hold the scrollable content
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Add a canvas to the content frame
        self.canvas = tk.Canvas(self.content_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a frame inside the canvas to hold the actual content
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")

        # Bind events to handle resizing and scroll region updates
        self.scrollable_frame.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._center_content)

        # Bind the mouse wheel to scroll the canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

        # Add content to the scrollable frame
        self.label = tk.Label(
            self.scrollable_frame,
            text="Before proceeding, it is important to understand the actions and movements required to participate in this experiment.",
        )
        self.label.pack(pady=20)

        self.label = tk.Label(self.scrollable_frame, text="Relax. Stand up straight with your arms to your sides.")
        self.label.pack(pady=2)

        self.label = tk.Label(self.scrollable_frame, text="We will refer to this stance as your Neutral Position.")
        self.label.pack(pady=20)

        # Load and display the first image
        self.display_image("assets/Neutral_Stance.jpeg")

        self.label = tk.Label(self.scrollable_frame, text="When prompted with a Red dot on screen: " + actions["left_elbow_flex"].text)
        self.label.pack(pady=20)

        # Load and display the second image
        self.display_image("assets/Left_Raise.jpeg")

        self.label = tk.Label(self.scrollable_frame, text="When prompted with a Blue dot on screen, bend the elbow of your Right arm to the horizontal position.")
        self.label.pack(pady=20)

        # Load and display the third image
        self.display_image("assets/Right_Raise.jpeg")

        self.label = tk.Label(self.scrollable_frame, text="When prompted with a Green dot on screen, bow forward by 20 degrees.")
        self.label.pack(pady=20)

        # Load and display the fourth image
        self.display_image("assets/Neutral_Stance.jpeg")

        # Button to proceed to InfoApp
        self.proceed_button = tk.Button(
            self.scrollable_frame, text="Ready for Trials", command=lambda: self.go_to_timer_app(option_num)
        )
        self.proceed_button.pack(pady=10)
        
        # Button to go back to OptionsPage
        self.back_button = tk.Button(
            self.scrollable_frame, text="Back to Options", command=self.go_to_options_page
        )
        self.back_button.pack(pady=10)

    def _update_scroll_region(self, event):
        """Update the scroll region to include the entire scrollable frame."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _center_content(self, event):
        """Center the content in the canvas when the window is resized."""
        canvas_width = event.width
        frame_width = self.scrollable_frame.winfo_reqwidth()
        if frame_width < canvas_width:
            self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        else:
            self.canvas.itemconfig(self.canvas_window, width=frame_width)

    def _on_mouse_wheel(self, event):
        """Scroll the canvas when the mouse wheel is moved."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def display_image(self, image_path):
        """Display an image inside the scrollable frame."""
        image = Image.open(image_path)  # Make sure the path is correct
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(self.scrollable_frame, image=photo)  # Create a label for the image
        image_label.image = photo  # Prevent image from being garbage-collected
        image_label.pack(pady=10)  # Pack the image label to make it visible

    def go_to_timer_app(self, option_num=0):
        """Destroy the current window and go to TimerApp."""
        self.root.destroy()
        timer_root = tk.Tk()
        TimerApp(timer_root, user_choice=option_num)  # Open TimerApp directly
        timer_root.mainloop()
    
    def go_to_options_page(self):
        """Destroy the current window and go to TimerApp."""
        self.root.destroy()
        dropdown_root = tk.Tk()  # Create a new Tk instance for OptionsPage
        OptionsPage(dropdown_root)  # Initialize OptionsPage
        dropdown_root.mainloop()


##########################
### First - Login Page ###
##########################
class LoginPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Login Page")
        
        # Set the window size to make it larger
        self.root.geometry("600x400")
        
        # Create a frame to hold the widgets and center it
        self.frame = tk.Frame(self.root)
        self.frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Load and add the transparent PNG above the entry box
        self.add_image()

        # Email label and entry
        self.email_label = tk.Label(self.frame, text="Email:")
        self.email_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        self.email_entry = tk.Entry(self.frame, relief="solid", highlightthickness=2)
        self.email_entry.grid(row=1, column=1, padx=10, pady=10)
        
        # Add focus bindings for email entry to highlight on click
        self.email_entry.bind("<FocusIn>", self.on_focus_in_email)
        self.email_entry.bind("<FocusOut>", self.on_focus_out_email)

        # Password label and entry
        self.password_label = tk.Label(self.frame, text="Password:")
        self.password_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")

        self.password_entry = tk.Entry(self.frame, show="*", relief="solid", highlightthickness=2)
        self.password_entry.grid(row=2, column=1, padx=10, pady=10)
        
        # Add focus bindings for password entry to highlight on click
        self.password_entry.bind("<FocusIn>", self.on_focus_in_password)
        self.password_entry.bind("<FocusOut>", self.on_focus_out_password)

        # Add a click event to the root to remove highlight
        self.root.bind("<Button-1>", self.remove_highlight)

        # Custom oval button
        self.create_oval_button()

    def add_image(self):
        """Loads a transparent PNG image and places it above the email entry."""
        image = Image.open("assets/NeurotechLogo.png")  # Replace with your image path
        self.photo = ImageTk.PhotoImage(image)
        
        # Create a label for the image
        self.image_label = tk.Label(self.frame, image=self.photo)
        self.image_label.grid(row=0, columnspan=2, pady=10)  # Place above email entry

    def on_focus_in_email(self, event):
        """Highlight the email entry when it gains focus."""
        self.email_entry.config(highlightbackground="#A0C7C3", highlightcolor="#A0C7C3", bg="#E8F6F3")

    def on_focus_out_email(self, event):
        """Remove highlight from the email entry when it loses focus."""
        self.email_entry.config(highlightbackground="gray", highlightcolor="gray", bg="white")

    def on_focus_in_password(self, event):
        """Highlight the password entry when it gains focus."""
        self.password_entry.config(highlightbackground="#A0C7C3", highlightcolor="#A0C7C3", bg="#E8F6F3")

    def on_focus_out_password(self, event):
        """Remove highlight from the password entry when it loses focus."""
        self.password_entry.config(highlightbackground="gray", highlightcolor="gray", bg="white")

    def remove_highlight(self, event):
        """Remove highlight from both entry fields when clicking anywhere else."""
        self.on_focus_out_email(event)
        self.on_focus_out_password(event)

    def create_oval_button(self):
        """Creates an oval-shaped button using a canvas with press animation."""
        # Place the button in row 3, below the password entry
        self.canvas = tk.Canvas(self.frame, width=200, height=50, highlightthickness=0)
        self.canvas.grid(row=3, columnspan=2, pady=20)  # Move to row 3

        # Create the oval button with tags for future reference
        self.oval_button = self.canvas.create_oval(10, 10, 190, 40, outline="#A0C7C3", fill="#A0C7C3", tags="oval_button")
        
        # Add text inside the button
        self.button_text = self.canvas.create_text(100, 25, text="Login", fill="white", font=("Arial", 12, "bold"), tags="oval_button")

        # Bind mouse click and release events for animation
        self.canvas.tag_bind("oval_button", "<ButtonPress-1>", self.on_button_press)
        self.canvas.tag_bind("oval_button", "<ButtonRelease-1>", self.on_button_release)


    def on_button_press(self, event):
        """Simulates button press by changing its color and slightly shifting the canvas."""
        self.canvas.itemconfig(self.oval_button, fill="#7CACAC")  # Darken color for "press"
        self.canvas.move(self.oval_button, 2, 2)  # Move slightly down-right
        self.canvas.move(self.button_text, 2, 2)  # Move text with the button

    def on_button_release(self, event):
        """Simulates button release by reverting the color and triggering the login function."""
        self.canvas.itemconfig(self.oval_button, fill="#A0C7C3")  # Restore original color
        self.canvas.move(self.oval_button, -2, -2)  # Move back to the original position
        self.canvas.move(self.button_text, -2, -2)  # Move text back with the button

        # Trigger the login logic after the release
        self.login()

    def login(self):
        email = self.email_entry.get()
        password = self.password_entry.get()

        correct_email = "neurotechatuiuc@gmail.com"
        correct_password = "2024NTX2015"

        if email == correct_email and password == correct_password:
            messagebox.showinfo("Login Successful", "Welcome!")
            self.root.destroy()  # Close the login window
            self.open_dropdown_page()  # Open InfoApp
        else:
            messagebox.showerror("Login Failed", "Incorrect email or password.")
    
    def open_dropdown_page(self):
        dropdown_root = tk.Tk()  # Create a new Tk instance for OptionsPage
        OptionsPage(dropdown_root)  # Initialize OptionsPage
        dropdown_root.mainloop()

def main():
    root = tk.Tk()
    app = LoginPage(root)
    root.mainloop()

if __name__ == "__main__":
    main()
