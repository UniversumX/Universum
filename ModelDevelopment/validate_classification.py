import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from utils import *

import matplotlib

matplotlib.use("TkAgg")


def validate_model(model, x, y, problem_type="classification"):
    """
    Validate that the model works on the given dataset and output relevant statistics and graphs.

    Args:
        model: Trained machine learning model with a `predict` method.
        x: Input features (numpy array or pandas DataFrame).
        y: True labels (numpy array or pandas Series).
        problem_type: Type of problem, either "classification" or "regression".

    Returns:
        metrics: Dictionary containing calculated metrics.
    """
    # Make predictions
    y_pred = model.predict(x)

    metrics = {}

    if problem_type == "classification":
        # Accuracy, Precision, Recall, F1 Score
        metrics["accuracy"] = accuracy_score(y, y_pred)
        metrics["precision"] = precision_score(y, y_pred, average="weighted")
        metrics["recall"] = recall_score(y, y_pred, average="weighted")
        metrics["f1_score"] = f1_score(y, y_pred, average="weighted")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # ROC Curve
        # if len(np.unique(y)) == 2:  # Binary classification
        #     y_prob = model.predict_proba(x)[:, 1]
        #     fpr, tpr, _ = roc_curve(y, y_prob)
        #     roc_auc = auc(fpr, tpr)
        #     plt.figure()
        #     plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        #     plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        #     plt.xlabel("False Positive Rate")
        #     plt.ylabel("True Positive Rate")
        #     plt.title("ROC Curve")
        #     plt.legend()
        #     plt.show()
        #
    elif problem_type == "regression":
        # MSE, MAE, R2 Score
        metrics["mse"] = mean_squared_error(y, y_pred)
        metrics["mae"] = mean_absolute_error(y, y_pred)
        metrics["r2_score"] = r2_score(y, y_pred)

        # Residual Plot
        residuals = y - y_pred
        plt.figure()
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def load_model(path, model_type):
    """
    Load a trained model from the given path.
    Args:
        path -- the path to the model
        model_type - the type of the model
            pkl - a pickle file, typically saved from sklearn
    """

    if model_type == "svm":
        import joblib

        model = joblib.load(path)
        return model

    raise ValueError(f"Invalid model type: {model_type}")


def visualize_model_classification(model, x):
    y_pred = model.predict(x)

    # Visualize the predictions
    plt.scatter(np.arange(len(x)), y_pred, label="Predicted")
    plt.xlabel("Index")
    plt.ylabel("Predicted Value")
    plt.show()


if __name__ == "__main__":
    # Validate the model
    model_path = "./models/110/1/svm_model.joblib"
    model = load_model(model_path, "svm")
    subject_id = "110"
    visit_number = "1"
    actions = {
        "left_elbow_flex": Action(
            action_value=1,
            text="Please flex your left elbow so your arm raises to shoulder level",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "left_elbow_relax": Action(
            action_value=2,
            text="Please relax your left elbow back to original state",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "right_elbow_flex": Action(
            action_value=3,
            text="Please flex your right elbow so your arm raises to shoulder level",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "right_elbow_relax": Action(
            action_value=4,
            text="Please relax your right elbow back to original state",
            audio="path/to/audio",
            image="path/to/image",
        ),
        "end_collection": Action(
            action_value=5, text="Data collection ended", audio=None, image=None
        ),
    }
    x, y = load_data_and_labels(subject_id, visit_number, actions)
    print(
        f"Validating model {model_path} on subject {subject_id}, visit {visit_number}"
    )
    validate_model(model, x, y, problem_type="classification")
    visualize_model_classification(model, x)
