import os
import cv2
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import load_model

# --------------------------
# LOAD MODEL
# --------------------------
model = load_model("/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN.h5")
print("Model loaded!")

# 🔒 FREEZE ALL PREVIOUS LAYERS EXCEPT THE LAST ONE
for layer in model.layers[:-1]:
    layer.trainable = False

# --------------------------
# CLASS MAPPING
# --------------------------
classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)

IMG_SIZE = 90

# --------------------------
# LOAD NEW DIGIT DATA ONLY
# --------------------------
new_data_dirs = [
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers/",
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers_data_gen/"
]

X_new = []
y_new = []

for data_dir in new_data_dirs:
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            char_label = filename.split("_")[0]  # digit string "0"-"9"

            img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0

            X_new.append(img)
            y_new.append(char_label)

X_new = np.array(X_new)
X_new = np.expand_dims(X_new, axis=-1)   # REQUIRED FOR CNN

y_new_int = np.array([char_to_idx[c] for c in y_new])
y_new_1hot = keras.utils.to_categorical(y_new_int, num_classes)

model.summary()

# --------------------------
# TRAINING SETTINGS
# --------------------------
epochs       = 10        # Use smaller epochs when freezing layers
batch_size   = 64
val_split    = 0.2

# LOWER LEARNING RATE FOR FINE-TUNING
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------
# TRAIN ONLY THE FINAL LAYER
# --------------------------
history = model.fit(
    X_new,
    y_new_1hot,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_split=val_split
)

# --------------------------
# SAVE FINE-TUNED MODEL
# --------------------------
model.save("/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN_newest1.h5")
