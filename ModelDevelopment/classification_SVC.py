import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from preprocessing import preprocess_person  # Import your preprocessing function
from scipy.stats import mode
from sklearn.decomposition import PCA
from typing import Dict
from utils import *
from dataclasses import dataclass

# Example of usage in load_data_and_labels
# Assuming `frequencies` is an array of frequency values corresponding to the third dimension of eeg_data
def load_data_and_labels(subject_id, visit_number, actions):
    # Load preprocessed data
    directory_path = f"../DataCollection/data/EEGdata/{subject_id}/{visit_number}/"
    # eeg_data, accel_data, action_data 
    res = preprocess_person(
        directory_path,
        actions,
        should_visualize=False,
    )
    eeg_feature_combined = []
    accel_data_combined = []
    action_data_combined = []
    for eeg_feature, accel_data, action_data in res:
        eeg_feature_combined.append(eeg_feature)
        accel_data_combined.append(accel_data)
        action_data_combined.append(action_data)
    
    # Merge all arrays using np.concatenate
    eeg_feature_combined = np.concatenate(eeg_feature_combined, axis=0)
    accel_data_combined = np.concatenate(accel_data_combined, axis=0)
    action_data_combined = np.concatenate(action_data_combined, axis=0)

    # Print the number of epochs in eeg_data
    num_epochs = len(eeg_feature_combined)
    print(f"Number of epochs in EEG data: {num_epochs}")
    
    # Print the number of entries in action_data
    num_action_entries = len(action_data_combined)
    print(f"Number of action data entries: {num_action_entries}")

    X = np.array(eeg_feature_combined)
    y = np.array(action_data_combined)

    return X, y


def classify_eeg_data(subject_id, visit_number, actions):
    # Load and preprocess data from multiple trials
    X, y = load_data_and_labels(subject_id, visit_number, actions) # X = [num_epoch,num_channel,eigvec, num_component]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Instantiate model and train
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X_train, y_train)

    # Determine accuraacy of model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save model to specific path
    model_dir = f"models/{subject_id}/{visit_number}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "svm_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    



if __name__ == "__main__":
    # Example actions dictionary (replace with actual Action objects)
    actions = {
        "left_elbow_relax": Action(
            action_value=1, text="Left Elbow Relax", audio="", image=""
        ),
        "left_elbow_flex": Action(
            action_value=2, text="Left Elbow Flex", audio="", image=""
        ),
        "right_elbow_relax": Action(
            action_value=3, text="Right Elbow Relax", audio="", image=""
        ),
        "right_elbow_flex": Action(
            action_value=4, text="Right Elbow Flex", audio="", image=""
        ),
        "end_collection": Action(
            action_value=5, text="End Collection", audio="", image=""
        ),
    }

    # Example parameters (replace with actual values)
    subject_id = "110"
    visit_number = 1

    classify_eeg_data(subject_id, visit_number, actions)
