import string
import pandas as pd
import cv2
from sklearn.utils import shuffle
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
print('Success')
print('Success')

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
    target_height = 90
    target_width = 90

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


model_path = "/home/fizzer/Downloads/clueboard_reader_model_v4.h5"  # change to your path
model = load_model(model_path)

classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 " # class 36 is just a space, ' '
char_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_char = {i: c for i, c in enumerate(classes)}

IMG_SIZE = 90

def predict_board(img_path):
    result = []

    words = extract_board_words(img_path)
    for word_idx, word in enumerate(words):
        chars = characterize_word(word)
        for char in chars:
            # Resize and normalize
            char_img = cv2.resize(char, (IMG_SIZE, IMG_SIZE))
            char_img_normal = char_img.astype("float32") / 255.0

            # Add batch dimension (and channel if needed)
            # crop_input = char_img_normal.reshape(1, 90, 90, 1)

            # Predict character
            prediction = model.predict([[char_img_normal]], verbose=0)
            char_idx = np.argmax(prediction, axis=1)[0]

            result.append(idx_to_char[char_idx])

        # Add a space after each word except the last one
        if word_idx < len(words) - 1:
            result.append(" ")

    return result

# Test on one image
img_path = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/validation_data/crime_1/PLACE_FOREST.png"

print("Predicted board:")
result = predict_board(img_path)

print("".join(result))