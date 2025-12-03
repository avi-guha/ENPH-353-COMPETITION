#!/usr/bin/env python3

import string
import pandas as pd
import cv2
from sklearn.utils import shuffle
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

print('Success')

IMG_SIZE = 90

classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 " # class 36 is just a space, ' '
char_to_idx = {c: i for i, c in enumerate(classes)}
print(char_to_idx)

idx_to_char = {i: c for i, c in enumerate(classes)}
print(idx_to_char)

X = []  # images array
y = []  # corresponding labels
IMG_SIZE = 90

data_dir = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_chars/"

for filename in os.listdir(data_dir):
    if filename.endswith(".png"):
        char_label = filename.split("_")[0]
        char_img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)# read in greyscale
        char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE))
        char_img_normal = char_img.astype("float32") / 255.0 # normalize

        X.append(char_img_normal)
        y.append(char_label)

X = np.array(X)

# convert to indices from our dictionary
y_int = np.array([char_to_idx[c] for c in y])


# convert to one hot
num_classes = len(classes)
y_total = keras.utils.to_categorical(y_int, num_classes=num_classes)

# shuffle for randomness
X_total, y_total = shuffle(X, y_total, random_state=42)

# Tune param as needed
epochs = 32
batch_size = 64
learning_rate = 0.001
val_split=0.2 # % of data to be used as validation

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

input_shape = (90, 90, 1)
num_classes = 37  # A-Z + 0-9 + space

model = Sequential([
    Input(shape=input_shape),  # <- Keras 3 style, NO batch_shape

    Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()


model.compile(optimizer='adam', # as discussed in class
              loss='categorical_crossentropy', # for multiple classes
              metrics=['accuracy'])

history = model.fit(X_total,
                    y_total,
                    batch_size=batch_size, # defined above
                    epochs=epochs, # defined above
                    verbose=1,
                    validation_split=val_split)

model.save("/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN_latest.h5")