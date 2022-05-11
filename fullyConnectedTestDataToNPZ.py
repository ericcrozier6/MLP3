import librosa as lb
import os
import numpy as np
import pandas as pd

#saves the signal data from each mp3 to a npz file.

test_data = np.empty((639450,), dtype=np.float32)

for fileName in os.scandir(os.path.abspath("./MusicData/test/")):
    data, sr = lb.load(os.path.abspath(fileName), duration=29)
    test_data = np.vstack((test_data, data))

test_data = np.delete(test_data, 0, axis=0)


sp.sparse.save_npz("fullyConnectedTrainData.npz", sp.sparse.csr_matrix(test_data))