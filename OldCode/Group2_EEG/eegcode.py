import neurosity
import sqlite3

# Connect to the Neurosity API
neurosity.connect()

# Set up the database connection
conn = sqlite3.connect("eeg_data.db")
cursor = conn.cursor()

# Create a table to store the EEG data
cursor.execute(
    "CREATE TABLE IF NOT EXISTS eeg_data (timestamp REAL, delta REAL, theta REAL, alpha REAL, beta REAL, gamma REAL)"
)

# Start streaming the EEG data
neurosity.start_stream(
    ["delta", "theta", "alpha", "beta", "gamma"],
    lambda data: cursor.execute(
        "INSERT INTO eeg_data VALUES (?, ?, ?, ?, ?, ?)",
        (data["timestamp"], data["delta"], data["theta"], data["alpha"], data["beta"], data["gamma"])
    )
)

# Run the stream for 60 seconds
import time
time.sleep(60)

# Stop the stream and close the database connection
neurosity.stop_stream()
conn.close()






# you will still need to write the bulk of the code.
#
#Here is an example of how the connection to the Neurosity Crown headset could be established using the Neurosity SDK in Python:

import neurosdk

# create a new session
session = neurosdk.Session()

# Connect to the headset
headset = session.connect('COM3')

#set the headset to record data from the parietal lobe
headset.set_electrode_config(['P4'])

#This code snippet establishes a connection to the Neurosity Crown headset that is connected to the computer's COM3 port, and sets the headset to record data from the parietal lobe.


# an example of how data could be collected and stored in a database using python and a mysql connector

import mysql.connector
from datetime import datetime

#connect to the database
cnx = mysql.connector.connect(user='user',password='password',host='hostname',database='databasename')
cursor = cnx.cursor()

while True:
    # Collect the neural data
    data = headset.get_data()
    timestamp = datetime.now()

    # insert data into the database
    insert_query = "INSERT INTO neural_data (data, timestamp) VALUES (%s, %s)"
    cursor.execute(insert_query, (data, timestamp))
    cnx.commit()


#This code snippet collects neural data from the parietal lobe and stores the data along with its timestamp in the 'neural_data' table of the connected MySQL database everytime the data is available.

#Please keep in mind that you will need to adjust this code to suit your specific requirements and you might need to do some more research on the functions and classes used here, to have a better understanding of how they work, and how they can be used.


