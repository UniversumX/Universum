

import cv2
import numpy as np
import openpose as op

# Initialize the OpenPose model
op_model = op.initialize_openpose(model_folder="openpose_models")

# Open the video file
video = cv2.VideoCapture("video.mp4")

# Loop through the frames of the video
while video.isOpened():
    # Read the current frame
    success, frame = video.read()
    if not success:
        break

    # Run the frame through the OpenPose model
    keypoints, output_image = op.forward(frame, op_model)

    # Extract the head keypoints from the keypoints array
    head_keypoints = keypoints[0]

    # Calculate the pitch, roll, and yaw from the head keypoints
    pitch = np.arctan2(head_keypoints[1], head_keypoints[2])
    roll = np.arctan2(head_keypoints[3], head_keypoints[4])
    yaw = np.arctan2(head_keypoints[5], head_keypoints[6])

    # Display the output image and the pitch, roll, and yaw values
    cv2.imshow("Output", output_image)
    print(f"Pitch: {pitch:.2f} Roll: {roll:.2f} Yaw: {yaw:.2f}")

    # Wait for the user to press a key before continuing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the window
video.release()
cv2.destroyAllWindows()


# the above was for head pry estimation, we need to change the key points for the torso

import cv2
import openpose as op
import numpy as np
import time
from datetime import datetime
import mysql.connector

# Step 1: Initialize OpenPose
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = "models"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Step 2: Create function to calculate roll, pitch, and yaw
def calculate_rpy(keypoints):
    # keypoints is an array of 25 body parts, including torso
    torso_keypoints = keypoints[2:6] # extract the keypoints of the torso
    x1, y1, c1 = torso_keypoints[0]
    x2, y2, c2 = torso_keypoints[1]
    x3, y3, c3 = torso_keypoints[2]
    x4, y4, c4 = torso_keypoints[3]
    # calculate the distances between the torso keypoints
    d1 = ((x2-x1)**2 + (y2-y1)**2)**0.5
    d2 = ((x3-x2)**2 + (y3-y2)**2)**0.5
    d3 = ((x4-x3)**2 + (y4-y3)**2)**0.5
    d4 = ((x4-x1)**2 + (y4-y1)**2)**0.5
    d5 = ((x3-x1)**2 + (y3-y1)**2)**0.5
    d6 = ((x3-x2)**2 + (y3-y2)**2)**0.5
    # use the distances to calculate roll, pitch, and yaw
    roll = np.arctan2(d1*d1+d2*d2-d3*d3-d4*d4, 2*d5*d6)
    pitch = np.arcsin(2*d5*d3/(d1*d1+d3*d3-d2*d2-d4*d4))
    yaw = np.arctan2(d1*d1-d2*d2+d3*d3-d4*d4, 2*d5*d6)
    return roll, pitch, yaw

# Step 3: Connect to your database
cnx = mysql.connector



