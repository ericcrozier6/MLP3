#code for our convolutional Neural Network, effectivly broken the size of the initial images is to large to compute for almost any machine.

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import PIL
import pathlib
import pandas as pd
import numpy as np

batch_size = 32
img_height = 1440
img_width = 1920
labels = pd.read_csv(os.path.abspath("./MusicData/train.csv")).sort_values(by='new_id').to_numpy()[:,1].tolist()

#path to the directory above the directory where the training spectrograms are stored
data_dir = os.path.abspath("./convoTrainData/")

#creates training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels = labels,
    validation_split = 0.2,
    subset = "training",
    seed=123,
    image_size = (img_height, img_width),
    batch_size = batch_size)

#creates validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=labels,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#deprecated would do the same thing as the rescaling layer in the model
norm_layer = tf.keras.layers.Rescaling(1./255)

#batch tuning
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#the convolution model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(1./255))
model.add(tf.keras.layers.Conv2D(2, (8, 8), strides=1, activation='relu', input_shape=(1440, 1920, 3)))
model.add(tf.keras.layers.Conv2D(3, (5, 5), strides=2, activation='relu'))
model.add(tf.keras.layers.Conv2D(5, (3, 3), strides=2, activation='relu'))
model.add(tf.keras.layers.Conv2D(8, (2, 2), strides=2, activation='relu'))
model.add(tf.keras.layers.Conv2D(13, (2, 2), strides=2, activation='relu'))

#start of dense layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10000, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.75))
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.75))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.75))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model
history = model.fit(train_ds, epochs=1, validation_data=val_ds)

#plot accuracy and relevant infor
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#create the testing dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.abspath("./convoTestData/"),
    labels = None,
    shuffle = False,
    image_size = (img_height, img_width),
    batch_size = batch_size)

#make and save our predictions
predictions = model.predict(test_ds)
numpy.savetxt("predictions.csv", predictions, delimiter=",", header=None)