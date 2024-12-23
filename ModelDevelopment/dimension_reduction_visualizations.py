from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import numpy as np
import pandas as pd
import preprocessing as pp
import numpy as np

# ------------------------------------------------------
# dimension reduction functions

# process data
def processresults(data): #used in ploting functions
    ## convert to power to avoid imaginary values
    data = np.abs(data) ** 2

    ## flatten data
    fdata = np.transpose(data, axes=(0, 1, 3, 2))
    fdata = fdata.reshape(-1, 18)

    #get epoch information
    epoch_indices = np.repeat(np.arange(data.shape[0]), data.shape[1] * data.shape[3]) #possible source of error, flattening may not arrange epochs like this
    epoch_indices = epoch_indices % 4 + 1
    return fdata, epoch_indices
    
# T-SNE
def plotWithTSNE(data_path, actions, isolate, action): #input data path, eg "../DataCollection/data/105/1/1/" actions done, boolean - whether or not to isolate action, which action to isolate
    eeg_data, acell_data, action_data = pp.preprocess(data_path, actions, False)
    data, epoch_indices = processresults(eeg_data)

    if isolate:
        indices = np.where(epoch_indices == action)[0]
        data = data[indices]
        epoch_indices = epoch_indices[indices]
    
    tsne = TSNE(n_components=2, perplexity=15, max_iter=1000, random_state=42) # change these values for different results
    tsne_embedding = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_embedding[:, 0], tsne_embedding[:, 1], c=epoch_indices, cmap="viridis", s=10
    )
    plt.colorbar(label="Action")
    plt.title(
        f"t-SNE projection of preprocessed data colored by action"
    )
    plt.show()


# UMAP
def plotWithUMAP(data_path, actions, isolate, action): #input data path, eg "../DataCollection/data/105/1/1/" actions done, boolean - whether or not to isolate action, which action to isolate
    eeg_data, acell_data, action_data = pp.preprocess(data_path, actions, False)
    data, epoch_indices = processresults(eeg_data)

    if isolate:
        indices = np.where(epoch_indices == action)[0]
        data = data[indices]
        epoch_indices = epoch_indices[indices]
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=epoch_indices, cmap="viridis", s=10)
    plt.colorbar(label="Action")
    plt.title(
        f"UMAP projection of preprocessed data colored by action"
    )
    plt.show()

def fixData(data_path): #doesn't work, doesn't line up with accelerometer data
    import os
    import datetime
    os.rename(data_path + "eeg_data_raw.csv", data_path + "old_eeg_data_raw.csv")
    input_file = data_path + "old_eeg_data_raw.csv"
    output_file = data_path + "eeg_data_raw.csv"

    data = pd.read_csv(input_file, header=None)

    columns = ["timestamp", "CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
    data.columns = columns

    reference_date = datetime.datetime(2024, 10, 17, 19, 39, 48, tzinfo=datetime.timezone.utc)
    data["timestamp"] = data["timestamp"].apply(
        lambda x: (reference_date + datetime.timedelta(seconds=x)).strftime("%Y-%m-%d %H:%M:%S.%f")
    )

    data.to_csv(output_file, index=False)

# ------------------------------------------------------

# Sample data
eeg_data_path = f"../DataCollection/data/EEGdata/108/1/1/"

from dataclasses import dataclass
@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str
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

plotWithUMAP(eeg_data_path, actions, False, 1)