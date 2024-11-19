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
    print(epoch_indices)
    return fdata, epoch_indices
    
# T-SNE
def plotWithTSNE(data_path, actions):
    eeg_data, acell_data, action_data = pp.preprocess(data_path, actions, False)
    data, epoch_indices = processresults(eeg_data)
    
    tsne = TSNE(n_components=2, perplexity=15, max_iter=1000, random_state=42) # change these values for different results
    tsne_embedding = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_embedding[:, 0], tsne_embedding[:, 1], c=epoch_indices, cmap="viridis", s=10
    )
    plt.colorbar(label="Action")
    plt.title(
        f"t-SNE projection of FFT-transformed whitened data colored by action"
    )
    plt.show()


# UMAP
def plotWithUMAP(data_path, actions): #input data path, eg "../DataCollection/data/105/1/1/" and actions
    eeg_data, acell_data, action_data = pp.preprocess(data_path, actions, False)
    data, epoch_indices = processresults(eeg_data)
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    print(embedding.shape)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=epoch_indices, cmap="viridis", s=10)
    plt.colorbar(label="Action")
    plt.title(
        f"UMAP projection of FFT-transformed whitened data colored by action"
    )
    plt.show()


# ------------------------------------------------------

# Sample data
eeg_data_path = f"../DataCollection/data/EEGdata/105/1/1/"

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

plotWithUMAP(eeg_data_path, actions)