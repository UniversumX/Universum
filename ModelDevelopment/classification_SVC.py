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


def classify_eeg_data(subject_id, visit_number, actions):
    # Load and preprocess data from multiple trials
    X, y = load_data_and_labels(
        subject_id, visit_number, actions
    )  # X = [num_epoch,num_channel,eigvec, num_component]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
