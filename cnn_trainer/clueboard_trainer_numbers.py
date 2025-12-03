import os
import cv2
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import load_model

model = load_model("/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN.h5")
print("Model loaded!")

classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)


IMG_SIZE = 90

new_data_dirs = [
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers/",
    "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers_data_gen/"
]

X_new = []
y_new = []

for data_dir in new_data_dirs:
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            char_label = filename.split("_")[0]  # e.g., "3"

            img = cv2.imread(os.path.join(data_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0    # same normalization

            X_new.append(img)
            y_new.append(char_label)

X_new = np.array(X_new)
X_new = np.expand_dims(X_new, axis=-1)
y_new_int = np.array([char_to_idx[c] for c in y_new])
y_new_1hot = keras.utils.to_categorical(y_new_int, num_classes)

model.summary()

epochs       = 32
batch_size   = 64
learning_rate = 0.001
val_split    = 0.2

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_new,
    y_new_1hot,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_split=val_split
)


model.save("/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN_newest.h5")
