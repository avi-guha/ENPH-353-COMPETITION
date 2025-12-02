#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import time
from std_msgs.msg import String
from board_reader import BoardReader
from PIL import Image as PImage


class BoardDetector:
    LOWER_BLUE_HSV = np.array([100, 50, 20])
    UPPER_BLUE_HSV = np.array([130, 255, 255])

    def __init__(self):
        self.cnn = BoardReader()
        self.bridge = CvBridge()

        self.last_board_time = time.time()
        self.last_yolo_time = 0

        self.current_board = 0

        # Expected board sequence
        self.board_map = {
            0: [True, "START"],
            1: [False, "SIZE"],
            2: [False, "VICTIM"],
            3: [False, "CRIME"],
            4: [False, "TIME"],
            5: [False, "PLACE"],
            6: [False, "MOTIVE"],
            7: [False, "WEAPON"],
            8: [False, "BANDIT"],
        }

        self.team_name = "teamEMAG"
        self.team_pass = "secret3"

        # --------- NEW CPU OPTIMIZATIONS ---------
        self.input_resize = (640, 360)    # Smaller YOLO input
        self.yolo_interval = 0.25         # YOLO every 0.25s (4 Hz)
        # throttle callback itself to avoid 30Hz overhead
        self.callback_interval = 0.05     # 20 Hz callback max
        self.last_callback_time = 0
        # ------------------------------------------

        rospy.init_node('yolo_clueboard_node')

        # Publishers
        self.pub_raw_board = rospy.Publisher('/clueboard/raw_board', Image, queue_size=1)
        self.pub_proc_board = rospy.Publisher('/clueboard/processed_board', Image, queue_size=1)
        self.pub_words_debug = rospy.Publisher('/clueboard/words_debug', Image, queue_size=1)
        self.pub_letters_debug = rospy.Publisher('/clueboard/letters_debug', Image, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        # Give BoardReader access
        self.cnn.bridge = self.bridge
        self.cnn.pub_words_debug = self.pub_words_debug
        self.cnn.pub_letters_debug = self.pub_letters_debug

        model_path = rospy.get_param(
            "~model_path",
            "/home/fizzer/ENPH-353-COMPETITION/src/competition/clueboard_detection/runs/detect/clueboards_exp12/weights/best.pt"
        )
        self.model = YOLO(model_path)

        # Fuse for small CPU gain
        try:
            self.model.fuse()
        except:
            pass

        self.camera_topic_left = rospy.get_param("~camera_topic_left",
                                                 "/B1/left_front_cam/left_front/image_raw")
        self.camera_topic_right = rospy.get_param("~camera_topic_right",
                                                  "/B1/right_front_cam/right_front/image_raw")

        rospy.Subscriber(self.camera_topic_left, Image, self.camera_callback_left, queue_size=1)
        rospy.Subscriber(self.camera_topic_right, Image, self.camera_callback_right, queue_size=1)

        rospy.loginfo("Board Detector Initialized!")

    # -----------------------------------------------------------
    #           CLEAN HELPER TO PROCESS LEFT/RIGHT CAMS
    # -----------------------------------------------------------
    def should_process_callback(self):
        """Drop ROS callback rate for CPU savings"""
        now = time.time()
        if now - self.last_callback_time < self.callback_interval:
            return False
        self.last_callback_time = now
        return True

    def try_yolo_now(self):
        """Rate-limit YOLO itself"""
        now = time.time()
        if now - self.last_yolo_time < self.yolo_interval:
            return False
        self.last_yolo_time = now
        return True

    # -----------------------------------------------------------
    #                 BOARD VALIDATION (unchanged)
    # -----------------------------------------------------------
    def board_captured(self, raw_board):
        hsv_image = cv2.cvtColor(raw_board, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.LOWER_BLUE_HSV, self.UPPER_BLUE_HSV)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.04 * peri, True)

        return 3 <= len(approx) <= 5

    # -----------------------------------------------------------
    #           LIGHTER process_raw_board for CPU
    # -----------------------------------------------------------
    def process_raw_board(self, raw_board):
        img = raw_board.copy()
        H, W = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (90, 60, 20), (130, 255, 255))
        cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img

        outer = max(cnts, key=cv2.contourArea)

        # Reduced Gaussian blur size (CPU optimization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # smaller blur kernel

        edges = cv2.Canny(blur, 12, 45)
        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [outer], -1, 255, -1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # smaller dilation kernel
        edges = cv2.dilate(edges, k, iterations=1)

        cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts2:
            return img

        inner = max(cnts2, key=cv2.contourArea)
        peri = cv2.arcLength(inner, True)
        approx = cv2.approxPolyDP(inner, 0.03 * peri, True)

        if len(approx) != 4:
            x, y, w, h = cv2.boundingRect(inner)
            return img[y:y + h, x:x + w]

        pts = approx.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        pts = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype=np.float32)

        w1 = np.linalg.norm(pts[0] - pts[1])
        w2 = np.linalg.norm(pts[3] - pts[2])
        h1 = np.linalg.norm(pts[0] - pts[3])
        h2 = np.linalg.norm(pts[1] - pts[2])
        W_out = int(max(w1, w2))
        H_out = int(max(h1, h2))

        dst = np.array([[0, 0], [W_out - 1, 0], [W_out - 1, H_out - 1], [0, H_out - 1]], np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        rectified = cv2.warpPerspective(img, M, (W_out, H_out))

        trim = 6
        if rectified.shape[0] > 2 * trim and rectified.shape[1] > 2 * trim:
            rectified = rectified[trim:-trim, trim:-trim]

        return rectified

    # -----------------------------------------------------------
    #           SHARED CAMERA PROCESSING (LEFT/RIGHT)
    # -----------------------------------------------------------
    def handle_frame(self, msg, forbidden_boards):
        """Shared heavy logic for left/right cameras"""
        # start timer
        if self.current_board == 0:
            out = String()
            out.data = f"{self.team_name},{self.team_pass},0,NA"
            self.pub_score.publish(out)
            self.board_map[0][0] = True
            self.current_board = 1
            return
        # final board
        if self.current_board > 8:
            out = String()
            out.data = f"{self.team_name},{self.team_pass},-1,NA"
            self.pub_score.publish(out)
            return

        # Drop callback FPS early
        if not self.should_process_callback():
            return

        # Board ordering logic
        if self.current_board in forbidden_boards:
            return
        if self.current_board not in self.board_map:
            return

        now = time.time()

        # Board cooldown (prevents duplicates)
        if now - self.last_board_time < 3.0:
            return

        # YOLO throttling BEFORE any cv_bridge or resize
        if not self.try_yolo_now():
            return

        # Only now convert the image (huge CPU savings)
        frame_full = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame_small = cv2.resize(frame_full, self.input_resize, interpolation=cv2.INTER_LINEAR)

        scale_x = frame_full.shape[1] / self.input_resize[0]
        scale_y = frame_full.shape[0] / self.input_resize[1]

        results = self.model.predict(frame_small, verbose=False)

        for r in results:
            for box in r.boxes:
                x1s, y1s, x2s, y2s = box.xyxy[0].tolist()
                conf = box.conf[0].item()

                x1, x2 = int(x1s * scale_x), int(x2s * scale_x)
                y1, y2 = int(y1s * scale_y), int(y2s * scale_y)

                sizeable = (x2 - x1) > 420
                ar = (x2 - x1) / max((y2 - y1), 1)
                confident = conf > 0.91

                ordered = (
                    (self.current_board - 1) in self.board_map and
                    self.board_map[self.current_board - 1][0] and
                    not self.board_map[self.current_board][0]
                )

                if sizeable and ar > 1.35 and confident and ordered:
                    frame_extract = frame_full[y1:y2, x1:x2]

                    if not self.board_captured(frame_extract):
                        return

                    roi = self.process_raw_board(frame_extract)
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    self.pub_raw_board.publish(self.bridge.cv2_to_imgmsg(frame_extract, "bgr8"))
                    self.pub_proc_board.publish(self.bridge.cv2_to_imgmsg(roi, "bgr8"))

                    pred_chars = self.cnn.predict_board(rgb)
                    words = "".join(pred_chars).split()
                    if not words:
                        return

                    predicted_label = words[0]
                    expected = self.board_map[self.current_board][1]

                    if predicted_label != expected:
                        return

                    text = "".join(w.replace(" ", "") for w in words[1:])
                    out = String()
                    out.data = f"{self.team_name},{self.team_pass},{self.current_board},{text}"
                    self.pub_score.publish(out)

                    self.board_map[self.current_board][0] = True
                    self.last_board_time = time.time()
                    self.current_board += 1
                    return

    # -----------------------------------------------------------
    #                  LEFT / RIGHT CALLBACKS
    # -----------------------------------------------------------
    def camera_callback_left(self, msg):
        self.handle_frame(msg, forbidden_boards=[2, 4, 6, 8])

    def camera_callback_right(self, msg):
        self.handle_frame(msg, forbidden_boards=[1, 3, 5, 7])


if __name__ == "__main__":
    try:
        detector = BoardDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
