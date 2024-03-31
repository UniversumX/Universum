import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the data
conn_eeg = sqlite3.connect("eeg_data.db")
conn_angles = sqlite3.connect("torso_angles.db")

eeg_data = pd.read_sql_query("SELECT * FROM eeg_data", conn_eeg)
torso_angles = pd.read_sql_query("SELECT * FROM angles", conn_angles)

conn_eeg.close()
conn_angles.close()

# Preprocess the data
eeg_data_grouped = eeg_data.pivot_table(index="timestamp", columns="channel", values="value").reset_index()
merged_data = torso_angles.merge(eeg_data_grouped, left_on="frame", right_on="timestamp", how="inner")
merged_data.drop(columns=["frame", "timestamp"], inplace=True)

# Normalize the data
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(merged_data), columns=merged_data.columns)

# Prepare input and output sequences
sequence_length = 10
n_features = len(normalized_data.columns) - 3

X = np.zeros((len(normalized_data) - sequence_length, sequence_length, n_features))
y = np.zeros((len(normalized_data) - sequence_length, 3))

for i in range(len(normalized_data) - sequence_length):
    X[i] = normalized_data.iloc[i:i + sequence_length, 3:].values
    y[i] = normalized_data.iloc[i + sequence_length, :3].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the LSTM model
model = Sequential([
    LSTM(units=50, activation="relu", input_shape=(sequence_length, n_features), return_sequences=True),
    LSTM(units=50, activation="relu"),
    Dense(units=3)
])

model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions
y_pred_inverse = scaler.inverse_transform(np.hstack([y_pred, np.zeros((y_pred.shape[0], n_features))]))[:, :3]
y_test_inverse = scaler.inverse_transform(np.hstack([y_test, np.zeros((y_test.shape[0], n_features))]))[:, :3]

# Compare the predicted angles with the ground truth
for i in range(5):
    print(f"Predicted angles: {y_pred_inverse[i]}, Actual angles: {y_test_inverse
