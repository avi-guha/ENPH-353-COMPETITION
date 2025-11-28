import string
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

print("Success")

# --- Parameters ---
TARGET_WIDTH = 600
TARGET_HEIGHT = 400
IMG_SIZE = 90

classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
char_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_char = {i: c for i, c in enumerate(classes)}

model_path = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN.h5"
model = load_model(model_path)

# --- Preprocessing ---
def preprocess_board(board_path):
    gray = cv2.imread(board_path, cv2.IMREAD_GRAYSCALE)
    gray_resized = cv2.resize(gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    gray_blur = cv2.medianBlur(gray_resized, 3)  # remove tiny noise
    gray_blur = cv2.GaussianBlur(gray_blur, (5, 5), 0)
    return gray_blur

# --- Extract words ---
def extract_board_words(board_path):
    gray = preprocess_board(board_path)
    board_height, board_width = gray.shape
    half_height = board_height // 2

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    # Morphological opening to remove tiny specks
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Split top and bottom
    img_top = binary[0:half_height, :]
    img_bottom = binary[half_height:, :]

    # Dilation to join letters into words
    kernel_top = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    kernel_bottom = cv2.getStructuringElement(cv2.MORPH_RECT, (35,5))
    dilated_top = cv2.dilate(img_top, kernel_top, iterations=2)
    dilated_bottom = cv2.dilate(img_bottom, kernel_bottom, iterations=2)

    # Find contours
    contours_top, _ = cv2.findContours(dilated_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_bottom, _ = cv2.findContours(dilated_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Shift bottom contours
    for cnt in contours_bottom:
        for point in cnt:
            point[0][1] += half_height

    all_contours = contours_top + contours_bottom

    # Filter contours by area and aspect ratio
    min_area = board_width * board_height * 0.005
    max_area = board_width * board_height * 0.5
    word_boxes = []
    for cnt in all_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h
        if area < min_area or area > max_area:
            continue
        if aspect_ratio < 0.5 or aspect_ratio > 10:
            continue
        word_boxes.append((x, y, w, h))

    # Define minimum dimensions for a word box to filter noise
    MIN_WORD_WIDTH = 30
    MIN_WORD_HEIGHT = 40
    
    filtered_boxes = []
    for x, y, w, h in word_boxes:
        
        # Filter: Ignore boxes that are too small (noise)
        if w < MIN_WORD_WIDTH or h < MIN_WORD_HEIGHT:
            continue
            
        filtered_boxes.append((x, y, w, h))

    word_boxes = filtered_boxes
    
    # Sort boxes by Y-coordinate (row) then X-coordinate (column) for correct reading order
    # The detective figure should now be the first box if it's in the top-left corner
    word_boxes = sorted(word_boxes, key=lambda b: (b[1], b[0]))
    
    # --- New Logic: Skip the first box (assuming it is the large detective figure) ---
    if len(word_boxes) > 0:
        word_boxes = word_boxes[1:]

    # Extract word images
    words = [gray[y:y+h, x:x+w] for x, y, w, h in word_boxes]

    # Debugging visualization
    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in word_boxes:
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return words

# --- Pad images ---
def pad_to_max(imgs, target_size=IMG_SIZE):
    padded = []
    for img in imgs:
        h, w = img.shape

        # If the image is larger than the target size, resize it down first
        if h > target_size or w > target_size:
            scale_factor = target_size / max(h, w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape
            
        top = (target_size - h) // 2
        bottom = target_size - h - top
        left = (target_size - w) // 2
        right = target_size - w - left
        
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        padded.append(padded_img)
    return padded

# --- Character extraction ---
def characterize_word(word_img):    
    # Use Adaptive Thresholding for better local contrast
    thresh = cv2.adaptiveThreshold(word_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological closing to connect broken character lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    letter_boxes = []
    vis_img = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR)
    
    # Estimate minimum character size based on the word image height
    h_word, w_word = word_img.shape
    MIN_CHAR_HEIGHT = h_word // 3 # Require minimum height to be 1/3 of the word image height
    MIN_CHAR_WIDTH = 5           # Require minimum width (must be wider than noise specks)
    
    # Filter contours based on size before sorting
    valid_contours = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        # Filter based on size
        if h >= MIN_CHAR_HEIGHT and w >= MIN_CHAR_WIDTH:
            valid_contours.append(ctr)
    
    contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        
        # Note: Size filtering is done above, but we extract here
        char_images.append(thresh[y:y+h, x:x+w])
        letter_boxes.append((x, y, w, h))
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Detect spaces
    if letter_boxes:
        # Check if any valid characters were actually extracted
        if not char_images:
            return []
            
        avg_char_width = np.mean([w for _, _, w, _ in letter_boxes])
        for i in range(len(letter_boxes)-1):
            x1, y1, w1, h1 = letter_boxes[i]
            x2, y2, w2, h2 = letter_boxes[i+1]
            gap = x2 - (x1 + w1)
            if gap > avg_char_width * 0.5:
                # Use the max height of the surrounding characters for the space
                h_space = max(h1, h2) if len(letter_boxes) > 1 else h1
                space_img = np.zeros((h_space, gap), dtype=word_img.dtype)
                char_images.insert(i+1, space_img)
                cv2.rectangle(vis_img, (x1 + w1, y1), (x2, y1 + h1), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return pad_to_max(char_images, target_size=IMG_SIZE)

# --- Prediction ---
def predict_board(img_path):
    result = []
    words = extract_board_words(img_path)
    for word_idx, word in enumerate(words):
        chars = characterize_word(word)
        for char in chars:
            char_img = cv2.resize(char, (IMG_SIZE, IMG_SIZE))
            char_img_normal = char_img.astype("float32") / 255.0
            char_img_input = char_img_normal.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            prediction = model.predict(char_img_input, verbose=0)
            char_idx = np.argmax(prediction, axis=1)[0]
            result.append(idx_to_char[char_idx])
        if word_idx < len(words) - 1:
            result.append(" ")
    return result

# --- Test ---
img_path = "/home/fizzer/ENPH-353-COMPETITION/src/clueboard_detection/yolo_inference_images/img_5.png"
#img_path = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/validation_data/crime_1/WEAPON_ICEBOMB.png"

print("Predicted board:")
result = predict_board(img_path)
print("".join(result))