from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import numpy as np
import pandas as pd
import preprocessing as pp

# ------------------------------------------------------
# dimension reduction functions
# T-SNE
def plotWithTSNE(data, coloring, colored_by):
    tsne = TSNE(n_components=2, perplexity=15, max_iter=1000, random_state=42)
    tsne_embedding = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_embedding[:, 0], tsne_embedding[:, 1], cmap="viridis", s=10
    )
    plt.colorbar(label="Timestamp")
    plt.title(
        f"t-SNE projection of FFT-transformed whitened data colored by {colored_by}"
    )
    plt.show()


# UMAP
def plotWithUMAP(data, coloring, colored_by):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    print(embedding.shape)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], cmap="viridis", s=10)
    plt.colorbar(label="Timestamp")
    plt.title(
        f"UMAP projection of FFT-transformed whitened data colored by {colored_by}"
    )
    plt.show()


# ------------------------------------------------------

# Sample data
eeg_data_path = f"../DataCollection/data/103/1/1/"

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

eeg_data, acell_data, action_data = pp.preprocess(eeg_data_path, actions, False)

print(eeg_data.shape)
## convert to power to avoid imaginary values
eeg_data = np.abs(eeg_data) ** 2

## flatten data
eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)

import numpy as np
eeg_data = eeg_data.T

print(eeg_data.shape)

plotWithTSNE(eeg_data, action_data, "none")
