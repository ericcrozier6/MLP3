import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy as sp

#our fully connected neural network using the raw signal data from the mp3s

batch_size = 32

#read in the training data from our npz
data = sp.sparse.load_npz("trainingMFCC.npz").toarray()

#create data and labels
train_data = data
train_labels = pd.read_csv(os.path.abspath("./MusicData/train.csv")).sort_values(by="new_id").to_numpy()[:,1]

#create train dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

#create test dataset
test_ds = tf.data.Dataset.from_tensor_slices(sp.sparse.load_npz("testingMFCC.npz").toarray())

train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

#our model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(rate=0.75),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.75),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dropout(rate=0.75),
    tf.keras.layers.Dense(6, activation='softmax')
])

#train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, epochs=10000)

#predict
predictions = model.predict(test_ds)
#save predictions
np.savetxt("fullConnectedNNPredictions.csv", predictions, delimiter=",")