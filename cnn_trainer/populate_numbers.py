#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === FIXED PATHS ===
OUTPUT_DIR = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers/"
AUG_DIR = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/training_numbers_data_gen/"
FONT_PATH = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/UbuntuMono-Regular.ttf"

IMG_SIZE = 90
FONT_SIZE = 72
IMAGES_PER_DIGIT = 50  # change this if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUG_DIR, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

print("Generating digit dataset...")

# === CLEAN DIGIT GENERATION ===
for digit in range(10):
    digit_str = str(digit)

    for i in range(IMAGES_PER_DIGIT):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Perfect text centering
        bbox = draw.textbbox((0, 0), digit_str, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (IMG_SIZE - text_w) // 2 - bbox[0]
        y = (IMG_SIZE - text_h) // 2 - bbox[1]

        draw.text((x, y), digit_str, fill=(0, 0, 0), font=font)

        filename = f"{digit_str}_{i+1}.png"
        img.save(os.path.join(OUTPUT_DIR, filename))

print(f"✔ Done! Generated {IMAGES_PER_DIGIT} clean images for each digit 0–9.")
print("Starting augmentation...")


# === AUGMENTATION FUNCTION ===
def datagen_img(img_path, num_to_gen):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = np.array(Image.open(img_path))

    datagen = ImageDataGenerator(
        rotation_range=5,
        zoom_range=0.09,
        brightness_range=[0.7, 1.5],
        shear_range=0.04,
        fill_mode="nearest"
    )

    image_array = np.expand_dims(img, 0)
    datagen_iterator = datagen.flow(image_array, batch_size=1)

    for j in range(num_to_gen):
        value = next(datagen_iterator)
        gen_img = value[0].astype("uint8")
        pil_img = Image.fromarray(gen_img)

        # Unique augmented filename
        filename = f"{img_name}_aug_{j}.png"
        file_path = os.path.join(AUG_DIR, filename)
        pil_img.save(file_path)


# === RUN AUGMENTATION ACROSS ALL CLEAN IMAGES ===
for img_file in os.listdir(OUTPUT_DIR):
    img_path = os.path.join(OUTPUT_DIR, img_file)
    datagen_img(img_path, 2)

print("✔ All augmentations complete!")
print(f"Clean images saved in: {OUTPUT_DIR}")
print(f"Augmented images saved in: {AUG_DIR}")
