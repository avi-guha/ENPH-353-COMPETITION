#!/usr/bin/env python3
import os
import cv2
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# -----------------------------
# DIRECTORY SETUP
# -----------------------------
LETTERS_DIR = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_chars/"   # already preprocessed
DIGIT_DIRS = [
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers/",
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers_data_gen/"
]

MODEL_OUTPUT_PATH = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN_retrained.h5"
IMG_SIZE = 90

# -----------------------------
# CLASS MAPPING
# -----------------------------
classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)


# =====================================================
# 1. LETTER LOADING  (NO preprocessing!)
# =====================================================
def load_preprocessed_letters(directory):
    X, Y = [], []

    if not os.path.isdir(directory):
        raise ValueError(f"Letters directory missing: {directory}")

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".png"):
            continue

        label = filename.split("_")[0]
        if label not in classes:
            print(f"[WARN] Invalid label '{label}' in file {filename}")
            continue

        path = os.path.join(directory, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Failed to read {path}")
            continue

        # Letters already 90×90 and binarized from your original pipeline
        img = img.astype("float32") / 255.0

        X.append(img)
        Y.append(label)

    return np.array(X), np.array(Y)


print("Loading PRE-PROCESSED letters (no preprocessing applied)...")
X_letters, y_letters = load_preprocessed_letters(LETTERS_DIR)
print("Loaded", X_letters.shape[0], "letters.")


# =====================================================
# 2. DIGIT LOADING  (WITH preprocessing)
# =====================================================
def preprocess_digit_img(gray):
    """Preprocess digits so they match letter preprocessing."""
    # Threshold → white char on black background
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    # Find character bounding box
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        crop = binary
    else:
        cnt = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
        x, y, w, h = cv2.boundingRect(cnt)
        crop = binary[y:y+h, x:x+w]

    # Pad to square
    h, w = crop.shape
    size = max(h, w)

    pad_top    = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left   = (size - w) // 2
    pad_right  = size - w - pad_left

    padded = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    # Resize → normalize
    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
    final = resized.astype("float32") / 255.0

    return final


def load_digit_images(dirs):
    X, Y = [], []

    for d in dirs:
        if not os.path.isdir(d):
            print(f"[WARN] Missing digit directory: {d}")
            continue

        for filename in os.listdir(d):
            if not filename.lower().endswith(".png"):
                continue

            label = filename.split("_")[0]  # Should be "0".."9"

            if label not in classes:
                print(f"[WARN] Invalid digit label '{label}' in {filename}")
                continue

            img_gray = cv2.imread(os.path.join(d, filename), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print("[WARN] Could not read:", filename)
                continue

            processed = preprocess_digit_img(img_gray)

            X.append(processed)
            Y.append(label)

    return np.array(X), np.array(Y)


print("Loading digits (with preprocessing)...")
X_digits, y_digits = load_digit_images(DIGIT_DIRS)
print("Loaded", X_digits.shape[0], "digits.")


# =====================================================
# 3. MERGE LETTERS + DIGITS
# =====================================================
X_total = np.concatenate([X_letters, X_digits], axis=0)
y_total = np.concatenate([y_letters, y_digits], axis=0)

# Shuffle
X_total, y_total = shuffle(X_total, y_total, random_state=42)

# Add channel dimension for CNN
X_total = np.expand_dims(X_total, axis=-1)  # (N, 90, 90, 1)

# Convert labels to one-hot
y_int = np.array([char_to_idx[c] for c in y_total])
y_cat = keras.utils.to_categorical(y_int, num_classes)


# =====================================================
# 4. TRAIN/VAL SPLIT
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_total, y_cat, test_size=0.2, random_state=42, stratify=y_int
)

print("Train samples:", X_train.shape[0])
print("Val samples:", X_val.shape[0])


# =====================================================
# 5. MODEL ARCHITECTURE
# =====================================================
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


model = build_model()
model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# 6. TRAIN
# =====================================================
history = model.fit(
    X_train, y_train,
    epochs=32,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)


# =====================================================
# 7. SAVE MODEL
# =====================================================
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
model.save(MODEL_OUTPUT_PATH)

print("✔ Model saved to:", MODEL_OUTPUT_PATH)
