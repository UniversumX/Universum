import cv2
import numpy as np
import sqlite3
from openpose import pyopenpose as op

def get_angles(V_shoulder, V_spine):
    V_up = np.cross(V_shoulder, V_spine)

    yaw = np.arctan2(V_shoulder[1], V_shoulder[0])
    pitch = np.arctan2(V_up[1], V_up[0])
    roll = np.arctan2(V_spine[1], V_spine[0])

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# Set up OpenPose
params = dict()
params["model_folder"] = "/path/to/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Set up SQLite database
conn = sqlite3.connect('torso_angles.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS angles (frame INTEGER, roll REAL, pitch REAL, yaw REAL)''')

# Process video
video_path = '/path/to/video.mp4'
cap = cv2.VideoCapture(video_path)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create OpenPose datum
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Extract relevant keypoints
    neck, r_shoulder, l_shoulder = datum.poseKeypoints[0, [1, 2, 5], :2]

    if len(neck) > 0 and len(r_shoulder) > 0 and len(l_shoulder) > 0:
        # Calculate vectors
        V_shoulder = l_shoulder - r_shoulder
        midpoint = (l_shoulder + r_shoulder) / 2
        V_spine = neck - midpoint

        # Calculate roll, pitch, and yaw
        roll, pitch, yaw = get_angles(V_shoulder, V_spine)

        # Store angles in the database
        c.execute("INSERT INTO angles (frame, roll, pitch, yaw) VALUES (?, ?, ?, ?)", (frame_id, roll, pitch, yaw))
        conn.commit()

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
conn.close()
