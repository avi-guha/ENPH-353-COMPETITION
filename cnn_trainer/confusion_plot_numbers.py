import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Load fine-tuned model
# -----------------------------
model = load_model("/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN_newest.h5")
print("Model loaded!")

# -----------------------------
# Class definitions (MUST MATCH TRAINING)
# -----------------------------
classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
digit_classes = "0123456789"
char_to_idx = {c: i for i, c in enumerate(classes)}

IMG_SIZE = 90

# -----------------------------
# Load digit test data
# -----------------------------
new_data_dirs = [
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers/",
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers_data_gen/"
]

X_test = []
y_true = []

for data_dir in new_data_dirs:
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            true_label = filename.split("_")[0]  # the digit

            img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0

            X_test.append(img)
            y_true.append(int(true_label))

X_test = np.array(X_test)
X_test = np.expand_dims(X_test, axis=-1)   # shape (N, 90, 90, 1)

y_true = np.array(y_true)

print("Loaded", X_test.shape[0], "digit samples.")

# -----------------------------
# Predict
# -----------------------------
y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert predicted class indices → digit 0–9
# Since digits start at index 26 in your class string, map back:
digit_start = classes.index("0")
y_pred_digits = y_pred - digit_start

# Clamp values outside 0–9 (in case model mispredicts a letter)
y_pred_digits = np.clip(y_pred_digits, 0, 9)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred_digits)
acc = accuracy_score(y_true, y_pred_digits)

print("Digit accuracy:", acc)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=digit_classes,
            yticklabels=digit_classes,
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Digit Accuracy = {acc:.3f})")
plt.show()
