import pandas as pd
import os

def label_eeg_data(data_path):
    """
    Add action labels to EEG data
    
    Parameters:
    data_path (str): Path to the folder containing EEG data file and action data file
    """
    file_path = os.path.join(data_path, "eeg_data_raw.csv")
    action_path = os.path.join(data_path, "action_data.csv")

    # Read the EEG data
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Read the Action data
    try:
        actions = pd.read_csv(action_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Create a new column for labels
    data['action_value'] = 0

    # Apply labels based on timestamp transitions
    for i in range(len(actions)-1):
        current_time = actions.iloc[i]['timestamp']
        next_time = actions.iloc[i+1]['timestamp']
        current_action = actions.iloc[i]['action_value']
        
        # Label all data points between current and next timestamp
        mask = (data['timestamp'] >= current_time) & (data['timestamp'] < next_time)
        data.loc[mask, 'action_value'] = current_action
    
    # Handle the last action (from last timestamp to end of data)
    last_time = actions.iloc[-1]['timestamp']
    last_action = actions.iloc[-1]['action_value']
    data.loc[data['timestamp'] >= last_time, 'action_value'] = last_action

    # Save the labeled data
    output_path = f"eeg_data_labeled.csv"
    data.to_csv(output_path, index=False)
    print(f"Labeled data saved to: {output_path}")

    return data

# ------------------------------------------------------

data_path = f"../DataCollection/data/EEGdata/103/1/1/"

labeled_data = label_eeg_data(data_path)