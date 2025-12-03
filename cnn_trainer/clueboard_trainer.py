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

def extract_board_words(board_path):
    """!
    @brief Returns a list of images of each word found in the input board image

    @param board_path: Path to the input board image-
    """
    gray = cv2.imread(board_path, cv2.IMREAD_GRAYSCALE)

    board_height, board_width = gray.shape  
    half_height = board_height//2

    #adaptive threshold - binarize img
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # split img into top / bottom
    img_top = binary[0:half_height, :]
    img_bottom = binary[half_height:board_height, :]

    # use different kernel for top / bottom
    # morphological dilation to join nearby characters into words
    kernel_top = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated_top = cv2.dilate(img_top, kernel_top, iterations=2)

    kernel_bottom = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    dilated_bottom = cv2.dilate(img_bottom, kernel_bottom, iterations=2)

    # Correctly unpack the result from cv2.findContours()
    contours_top, _ = cv2.findContours(dilated_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_bottom, _ = cv2.findContours(dilated_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # I only want to use second and third contours
    sorted_contours_left = sorted(contours_top, key=lambda c: cv2.boundingRect(c)[0])
    all_contours_top = sorted_contours_left[1:]

    list_top_bottom_word = []

    # debugging
    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for cnt in all_contours_top:
        x, y, w, h = cv2.boundingRect(cnt)
        list_top_bottom_word.append(gray[y:y+h, x:x+w])

        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for cnt in contours_bottom:
        x, y, w, h = cv2.boundingRect(cnt)
        y += half_height  # shift coordinates back to original image
        list_top_bottom_word.append(gray[y:y+h, x:x+w])

        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)


    #plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()

    return list_top_bottom_word


def pad_to_max(imgs):
    """
    Pads each image in `imgs` to a fixed size of 90x90 pixels.
    Images are centered and padded with black borders.
    """
    target_height = IMG_SIZE
    target_width = IMG_SIZE

    imgs_padded = []
    for img in imgs:
        height, width = img.shape

        # Compute equal padding on each side
        top = (target_height - height) // 2
        bottom = target_height - height - top

        left = (target_width - width) // 2
        right = target_width - width - left

        padded = cv2.copyMakeBorder(
            img,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        imgs_padded.append(padded)

    return imgs_padded


def characterize_word(word_img):
    """!
    @brief Breaks a word image into its constituent characters, including spaces

    @param word_img: A word (gray img) from the result of extract_board_words
    @return: A list of [INVERTED COLOUR] character images containing all characters in the word, including spaces
    """
    vis_img = word_img.copy()

    # threshold the image
    _, thresh = cv2.threshold(word_img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours (eachA contour is a letter)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left-to-right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Crop each letter
    char_images = []
    letter_boxes = []  # temp store coordinates for spacing
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        letter_img = thresh[y:y+h, x:x+w]
        char_images.append(letter_img)
        letter_boxes.append((x, y, w, h))
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # detect spaces by looking at distances between boxes
    for i in range(len(letter_boxes) - 1):
        x1, y1, w1, h1 = letter_boxes[i]
        x2, y2, w2, h2 = letter_boxes[i+1]
        gap = x2 - (x1 + w1)
        if gap > w1 * 0.5:  # threshold to consider as a space
            # create a blank white image for the space
            space_img = np.zeros((h1, gap), dtype=word_img.dtype)  # black instead of white
            char_images.insert(i+1, space_img)  # insert at correct position
            cv2.rectangle(vis_img, (x1 + w1, y1),
                          (x1 + w1 + gap, y1 + h1), (0, 0, 255), 2)  # red for spaces

    # Display the image with bounding boxes
    #plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()

    padded_char_images = pad_to_max(char_images)

    return padded_char_images


# TRAINING:
classes = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" # class 36 is just a space, ' '
char_to_idx = {c: i for i, c in enumerate(classes)}
print(char_to_idx)

idx_to_char = {i: c for i, c in enumerate(classes)}
print(idx_to_char)

X = []  # images array
y = []  # corresponding labels
IMG_SIZE = 90

data_dir = os.path.expanduser('~/ENPH-353-COMPETITION/cnn_trainer/training_chars/')

for filename in os.listdir(data_dir):
    if filename.endswith(".png"):
        char_label = filename.split("_")[0]
        char_img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)# read in greyscale
        char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE))
        char_img_normal = char_img.astype("float32") / 255.0 # normalize

        X.append(char_img_normal)
        y.append(char_label)

# Load images from training_numbers and training_numbers_data_gen directories
# These contain digit images with format: digit_number.png or digit_number_aug_index.png
numbers_dirs = [
    os.path.expanduser('~/ENPH-353-COMPETITION/cnn_trainer/training_numbers/'),
    os.path.expanduser('~/ENPH-353-COMPETITION/cnn_trainer/training_numbers_data_gen/')
]

for numbers_dir in numbers_dirs:
    if os.path.exists(numbers_dir):
        for filename in os.listdir(numbers_dir):
            if filename.endswith(".png"):
                # Label is the first part before underscore (e.g., "0" from "0_1.png" or "0_1_aug_0.png")
                char_label = filename.split("_")[0]
                char_img = cv2.imread(os.path.join(numbers_dir, filename), cv2.IMREAD_GRAYSCALE)
                char_img = cv2.resize(char_img, (IMG_SIZE, IMG_SIZE))
                # Invert colors: number images have white background/black text,
                # but training_chars has black background/white text
                char_img = 255 - char_img
                char_img_normal = char_img.astype("float32") / 255.0
                
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

input_shape=(IMG_SIZE , IMG_SIZE, 1) # save input shape as var

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

save_path = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN_ag.h5"
model.save(save_path)
print(f"Model saved to {save_path}")
