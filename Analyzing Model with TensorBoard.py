import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard

name = "cat-vs-dog-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(name))

pickle_in = open("x.pickle", "rb")
x = pickle.load(pickle_in) # pickle files created on Day 2

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)


x = np.array(x)
y = np.array(y)

x = x/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = x.shape[1:])) # First Layer
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3))) # Second Layer
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Third Layer
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1)) # Output Layer
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard])