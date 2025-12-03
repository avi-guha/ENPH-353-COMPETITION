#!/usr/bin/env python3
import string
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class BoardReader:
    """
    @class BoardReader
    @brief Reads and processes images of clue boards using our trained CNN model.

    This class loads our pre-trained Keras CNN model and provides encapsulated utility
    for board reading.
    """
    # Class parameters
    TARGET_WIDTH = 600
    TARGET_HEIGHT = 400
    IMG_SIZE = 90

    classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    char_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_char = {i: c for i, c in enumerate(classes)}

    # Constructor
    def __init__(self):
        model_path = "/home/fizzer/ENPH-353-COMPETITION/cnn_trainer/clueboard_reader_CNN_newest1.h5"
        self.model = load_model(model_path)

    # Preprocessing
    def preprocess_board(self, board):
        """
        @brief Preprocess cropped board to remove noise.

        @param board: RGB board image
        @return preprocessed grayscale board image
        """
        gray = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(
            gray,
            (self.TARGET_WIDTH, self.TARGET_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )
        gray_blur = cv2.medianBlur(gray_resized, 3)  # remove tiny noise
        # gray_blur = cv2.GaussianBlur(gray_blur, (5, 5), 0)
        return gray_blur

    def extract_board_words(self, board):
        gray = self.preprocess_board(board)
        board_height, board_width = gray.shape
        half_height = board_height // 2

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15, 10
        )

        # Morphological opening
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Split top/bottom
        img_top = binary[0:half_height, :]
        img_bottom = binary[half_height:, :]

        # Dilation
        kernel_top = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        kernel_bottom = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
        dilated_top = cv2.dilate(img_top, kernel_top, iterations=2)
        dilated_bottom = cv2.dilate(img_bottom, kernel_bottom, iterations=2)

        # Find contours
        contours_top, _ = cv2.findContours(
            dilated_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours_bottom, _ = cv2.findContours(
            dilated_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Shift bottom contours into full image coordinates
        for cnt in contours_bottom:
            for pt in cnt:
                pt[0][1] += half_height

        # -------------------------------------------------------
        # Identify the detective: tallest contour in top half
        # -------------------------------------------------------
        tallest_top = None
        max_h = 0
        for cnt in contours_top:
            _, _, _, h = cv2.boundingRect(cnt)
            if h > max_h:
                max_h = h
                tallest_top = cnt

        # Combine all contours
        all_contours = contours_top + contours_bottom

        # Area thresholds
        min_area = board_width * board_height * 0.005
        max_area = board_width * board_height * 0.5

        word_boxes = []

        # -------------------------------------------------------
        # Build word boxes, skipping only the detective
        # -------------------------------------------------------
        for cnt in all_contours:

            # Skip detective contour
            if cnt is tallest_top:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / h

            # Basic filters
            if area < min_area or area > max_area:
                continue
            if aspect_ratio < 0.5 or aspect_ratio > 10:
                continue

            # Word size filtering
            MIN_WORD_HEIGHT = 50
            MAX_WORD_HEIGHT = 250
            MIN_WORD_WIDTH = 30

            if w < MIN_WORD_WIDTH:
                continue
            if h < MIN_WORD_HEIGHT or h > MAX_WORD_HEIGHT:
                continue

            word_boxes.append((x, y, w, h))

        # Sort into reading order
        word_boxes = sorted(word_boxes, key=lambda b: (b[1], b[0]))

        # Extract word images
        words = [gray[y:y+h, x:x+w] for x, y, w, h in word_boxes]

        # Publish debug
        if hasattr(self, 'pub_words_debug'):
            debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for (x, y, w, h) in word_boxes:
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
            try:
                msg = self.bridge.cv2_to_imgmsg(debug, "bgr8")
                self.pub_words_debug.publish(msg)
            except:
                pass

        return words

    # Pad images
    def pad_to_max(self, imgs, target_size):
        """
        @brief Pads a list of images such that all are of dimension IMG_SIZE X IMG_SIZE.

        @param imgs: List of images to pad to constant size.
        @return List of input images, in same order, padded to constant size.
        """
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

            padded_img = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            padded.append(padded_img)
        return padded

    # Character extraction
    def characterize_word(self, word_img):
        """
        @brief Extracts individual chars (including spaces) from an image of a word.

        @param word_img: Image of word to extract char from.
        @return List of characters (inc. spaces) found in word, in image format.
        """
        # Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            word_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological closing to connect broken lines
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        char_images = []
        letter_boxes = []
        vis_img = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR)

        # Estimate minimum character size
        h_word, w_word = word_img.shape
        MIN_CHAR_HEIGHT = h_word // 3
        MIN_CHAR_WIDTH = 5

        # Filter contours based on size before sorting
        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if h >= MIN_CHAR_HEIGHT and w >= MIN_CHAR_WIDTH:
                valid_contours.append(ctr)

        contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            char_images.append(thresh[y:y+h, x:x+w])
            letter_boxes.append((x, y, w, h))
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Detect spaces
        if letter_boxes:
            if not char_images:
                return []

            avg_char_width = np.mean([w for _, _, w, _ in letter_boxes])
            for i in range(len(letter_boxes)-1):
                x1, y1, w1, h1 = letter_boxes[i]
                x2, y2, w2, h2 = letter_boxes[i+1]
                gap = x2 - (x1 + w1)
                if gap > avg_char_width * 0.5:
                    h_space = max(h1, h2) if len(letter_boxes) > 1 else h1
                    space_img = np.zeros((h_space, gap), dtype=word_img.dtype)
                    char_images.insert(i+1, space_img)
                    cv2.rectangle(vis_img, (x1 + w1, y1), (x2, y1 + h1), (0, 0, 255), 2)

        # Publish to GUI
        if hasattr(self, 'pub_letters_debug'):
            dbg = cv2.cvtColor(word_img, cv2.COLOR_GRAY2BGR)
            for (x, y, w, h) in letter_boxes:
                cv2.rectangle(dbg, (x, y), (x+w, y+h), (255, 0, 0), 2)
            try:
                msg = self.bridge.cv2_to_imgmsg(dbg, "bgr8")
                self.pub_letters_debug.publish(msg)
            except:
                pass

        return self.pad_to_max(char_images, self.IMG_SIZE)

    # --- Prediction ---
    def predict_board(self, img):
        """
        @brief Returns the model's prediction of clueboard ('img') as a list of chars.

        @param img: RGB format image of clueboard
        @return List of characters (inc. spaces) found in clueboard.
        """
        result = []
        words = self.extract_board_words(img)
        for word_idx, word in enumerate(words):
            chars = self.characterize_word(word)
            for char in chars:
                char_img = cv2.resize(char, (self.IMG_SIZE, self.IMG_SIZE))
                char_img_normal = char_img.astype("float32") / 255.0
                char_img_input = char_img_normal.reshape(
                    1, self.IMG_SIZE, self.IMG_SIZE, 1
                )
                prediction = self.model.predict(char_img_input, verbose=0)
                char_idx = np.argmax(prediction, axis=1)[0]
                result.append(self.idx_to_char[char_idx])
            if word_idx < len(words) - 1:
                result.append(" ")
        return result


if __name__ == "__main__":
    # Load image in RGB (as YOLO would output)
    img_path = "/home/fizzer/ENPH-353-COMPETITION/src/clueboard_detection/yolo_inference_images/img_14.png"
    board = cv2.cvtColor(
        cv2.imread(img_path, cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB
    )

    reader = BoardReader()

    print("Predicted board:")
    result = reader.predict_board(board)
    print("".join(result))
