## Procedure

1. Open up browser and go to the [website](https://console.neurosity.co/dashboard)
2. Login with the email address and password provided.
3. Scroll down to the bottom of the page and click on the "star" button of "signal quality" panel.
4. Make sure the signal quality is good.
5. If the signal quality is not good, try to adjust the headset and make sure the headset is in the correct position.
6. Using the phone app for a more detailed signal quality check and adjustment is also recommended.
7. Next, go to the "brainwaves" tab. you can also go to this [website](https://console.neurosity.co/brainwaves)
8. Click on the first "start" button. this will start the visualization of the brainwaves.
   After setup is complete, we will start the python script to record the data.
   The python script will record the data and save it to a file.
9. Open up a terminal and navigate to the directory where the python script is located.
10. Run the script:

`cd /path/to/Universum/DataCollection/`
`python3 gui.py`

This script will open a window asking for setup information (subject information, trial number, default trial time).
subject information includes:

- subject id
- subject visit number

Trial number is the current number of the trial that is being recorded. default is 0!!! 0 is for baseline data.

**_ NOTE: the default trial number is 0!!! 0 is for baseline data. _**

**_ NOTE: MAKE SURE YOU READ THE NOTES AND RECORD BASELINE DATA. _**

default trial time is the time in seconds that the trial will last. default is 60 seconds.

fill in the information and click "start" to start the recording.
another window will open with simple button controls.
click "start" to start the recording.
click "stop" to stop the recording.
click "reset" to reset the recording. NOTE: this will delete the current recording if remaining time > 0.
click "info" to show quick info of the headset in terminal.
click "discard last trial" to delete the last trial recorded. NOTE: this will delete the last trial recorded no matter of the remaining time. operate with caution.

after the recording is complete, close the window. the data are saved in the "data" folder in the same directory as the python script.

we will need to record sitting still for base line data.
after that we will record the data for tilting tasks.

ideally subjects should be in a quiet room with no distractions.
they should be sitting in a chair with their feet flat on the floor.
they should be sitting up straight with their back against the chair.
they should be looking straight ahead, maybe concentrate on a point on the wall, or monitor.

we should also record an eye tracking task so we have the baseline for eye movements when they are looking at the dot.
