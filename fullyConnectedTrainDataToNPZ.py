import librosa as lb
import os
import numpy as np
import pandas as pd

#saves the signal data from each mp3 in the training dataset to an npz file

train_data = np.empty((639450,), dtype=np.float32)

for fileName in os.scandir(os.path.abspath("./MusicData/train/")):
    data, sr = lb.load(os.path.abspath(fileName), duration=29)
    train_data = np.vstack((train_data, data))

train_data = np.delete(train_data, 0, axis=0)

train_labels = pd.read_csv(os.path.abspath("./MusicData/train.csv")).sort_values(by="new_id").to_numpy()[:,1].tolist()

train_data = np.append(train_data, train_labels, axis=1)

sp.sparse.save_npz("fullyConnectedTrainData.npz", sp.sparse.csr_matrix(train_data))