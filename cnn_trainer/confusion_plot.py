#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow import keras


# ================================
# SETTINGS
# ================================
MODEL_PATH = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN.h5"
TRAIN_DIR = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_chars/"
IMG_SIZE = 90
CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
VAL_SPLIT = 0.2  # still used for recreating splits, but not needed here


# ================================
# LOAD TRAINING DATA (REBUILD)
# ================================
def load_training_data():
    char_to_idx = {c: i for i, c in enumerate(CLASSES)}

    X = []
    y = []

    for filename in os.listdir(TRAIN_DIR):
        if filename.endswith(".png"):
            char_label = filename.split("_")[0]
            img = cv2.imread(os.path.join(TRAIN_DIR, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0

            X.append(img)
            y.append(char_label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_int = np.array([char_to_idx[c] for c in y])
    y_onehot = keras.utils.to_categorical(y_int, num_classes=len(CLASSES))

    # Use same deterministic shuffle to match training distribution
    X_shuf, y_shuf = shuffle(X, y_onehot, random_state=42)

    return X_shuf, y_shuf


# ================================
# CONFUSION MATRIX FOR ALL DATA
# ================================
def plot_confusion_matrix_all_data(model, X_all, y_all):
    y_pred_prob = model.predict(X_all)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_all, axis=1)

    # 37 classes, include missing as zero-rows
    all_labels = list(range(len(CLASSES)))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    plt.figure(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(CLASSES))
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title("Confusion Matrix – All Training Data (37 classes)")
    plt.tight_layout()
    plt.show()


# ================================
# MAIN
# ================================
def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading ALL training data...")
    X_total, y_total = load_training_data()

    print("Plotting confusion matrix for ALL CHARACTERS...")
    plot_confusion_matrix_all_data(model, X_total, y_total)


if __name__ == "__main__":
    main()

