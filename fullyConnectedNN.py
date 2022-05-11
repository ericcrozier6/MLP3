import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy as sp

#our fully connected neural network using the raw signal data from the mp3s

batch_size = 32

#read in the training data from our npz
data = sp.sparse.load_npz("fullyConnectedTrainData.npz").toarray()

#seperate data and labels
train_data = data[:, :len(data)]
train_labels = data[:, len(data)]

#create train dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

#create test dataset
test_ds = tf.data.Dataset.from_tensor_slices(sp.sparse.load_npz("fullyConnectedTestData.npz"))

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
model.fit(train_ds, epochs=10)

#predict
predictions = model.predict(test_ds)
#save predictions
np.savetxt("fullConnectedNNPredictions.csv", predictions, delimiter=",", header=None)