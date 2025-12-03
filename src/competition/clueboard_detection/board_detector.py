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


class BoardDetector:
    LOWER_BLUE_HSV = np.array([100, 50, 20])
    UPPER_BLUE_HSV = np.array([130, 255, 255])

    def __init__(self):
        self.cnn = BoardReader()
        self.bridge = CvBridge()

        self.last_board_time = time.time()
        self.current_board = 0
        self.frame_skip = 0

        # Detection sequence
        self.board_map = {
            0: [True,  "START"],
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

        rospy.init_node('yolo_clueboard_node')

        # Debug visualization publishers
        self.pub_raw_board = rospy.Publisher('/clueboard/raw_board', Image, queue_size=1)
        self.pub_proc_board = rospy.Publisher('/clueboard/processed_board', Image, queue_size=1)
        self.pub_words_debug = rospy.Publisher('/clueboard/words_debug', Image, queue_size=1)
        self.pub_letters_debug = rospy.Publisher('/clueboard/letters_debug', Image, queue_size=1)

        # Provide debug publishers to CNN
        self.cnn.bridge = self.bridge
        self.cnn.pub_words_debug = self.pub_words_debug
        self.cnn.pub_letters_debug = self.pub_letters_debug

        # Load YOLO
        model_path = rospy.get_param(
            "~model_path",
            "/home/fizzer/ENPH-353-COMPETITION/src/competition/clueboard_detection/runs/detect/clueboards_exp12/weights/best.pt"
        )
        self.model = YOLO(model_path)
        self.model.fuse = lambda *a, **k: self.model

        # Camera topics
        self.camera_topic_left  = rospy.get_param("~camera_topic_left",  "/B1/left_front_cam/left_front/image_raw")
        self.camera_topic_right = rospy.get_param("~camera_topic_right", "/B1/right_front_cam/right_front/image_raw")

        rospy.Subscriber(self.camera_topic_left,  Image, self.camera_callback, queue_size=1)
        rospy.Subscriber(self.camera_topic_right, Image, self.camera_callback, queue_size=1)

        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        rospy.loginfo("SUPER-Optimized Board Detector initialized.")

    # Check if whole board captured
    def board_captured(self, raw_board):
        hsv = cv2.cvtColor(raw_board, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE_HSV, self.UPPER_BLUE_HSV)
        return cv2.countNonZero(mask) > 2000

    # Extract homography
    def process_raw_board(self, raw):
        img = raw.copy()
        H, W = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (90,60,20), (130,255,255))
        cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img

        outer = max(cnts, key=cv2.contourArea)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [outer], -1, 255, -1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)

        cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts2:
            return img

        inner = max(cnts2, key=cv2.contourArea)
        peri = cv2.arcLength(inner, True)
        approx = cv2.approxPolyDP(inner, 0.03 * peri, True)

        if len(approx) != 4:
            x,y,w,h = cv2.boundingRect(inner)
            return img[y:y+h, x:x+w]

        pts = approx.reshape(4,2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        pts = np.array([
            pts[np.argmin(s)],     
            pts[np.argmin(diff)],  
            pts[np.argmax(s)],     
            pts[np.argmax(diff)],  
        ], np.float32)

        w1 = np.linalg.norm(pts[0] - pts[1])
        w2 = np.linalg.norm(pts[3] - pts[2])
        h1 = np.linalg.norm(pts[0] - pts[3])
        h2 = np.linalg.norm(pts[1] - pts[2])

        SCALE = 2.8
        W_out = int(max(w1, w2) * SCALE)
        H_out = int(max(h1, h2) * SCALE)

        dst = np.array([[0,0], [W_out-1,0], [W_out-1,H_out-1], [0,H_out-1]], np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        rectified = cv2.warpPerspective(img, M, (W_out, H_out), flags=cv2.INTER_CUBIC)

        # Trim border
        if rectified.shape[0] > 12 and rectified.shape[1] > 12:
            rectified = rectified[6:-6, 6:-6]

        return rectified
    

    # Camera callback for inference and reading
    def camera_callback(self, msg):
        # Starting case
        if self.current_board == 0:
            self.pub_score.publish(String(f"{self.team_name},{self.team_pass},0,NA"))
            self.current_board = 1
            return

        # cooldown
        if time.time() - self.last_board_time < 1.5:
            return

        # Skip every other frame for compute
        self.frame_skip = (self.frame_skip + 1) % 3
        if self.frame_skip != 0:
            return

        # Only run YOLO if board NOT acquired 
        #if self.board_map[self.current_board][0]:
            #return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # YOLO Inference (light mode) 
        results = self.model.predict(frame, verbose=False, imgsz=320)

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                bw = x2 - x1
                bh = y2 - y1

                # Cheap box filters
                sizeable = bw > 230
                aspect_ok = bw / bh > 1.0
                confident = conf > 0.65

                #prev_ok  = self.board_map[self.current_board - 1][0]
                #curr_not_done = not self.board_map[self.current_board][0]

                if not (sizeable and aspect_ok and confident): # remove prev_ok check, curr_not_done check
                    continue

                crop = frame[y1:y2, x1:x2]
                if not self.board_captured(crop):
                    continue

                roi = self.process_raw_board(crop)
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                self.pub_raw_board.publish(self.bridge.cv2_to_imgmsg(crop, "bgr8"))
                self.pub_proc_board.publish(self.bridge.cv2_to_imgmsg(roi, "bgr8"))

                chars = self.cnn.predict_board(rgb)
                words = "".join(chars).split()
                if not words:
                    return

                label = words[0]

                # Determine which board it is
                board_id = None
                for i, (_, name) in self.board_map.items():
                    if name == label:
                        board_id = i
                        break

                if board_id is None:
                    rospy.logwarn(f"Unknown board label: {label}")
                    return

                # If board already reported, skip
                if self.board_map[board_id][0]:
                    return

                # Extract remaining info
                rest = "".join(w.replace(" ", "") for w in words[1:])

                # Publish
                self.pub_score.publish(String(f"{self.team_name},{self.team_pass},{board_id},{rest}"))

                # Mark board as processed
                self.board_map[board_id][0] = True
                self.last_board_time = time.time()

                # If BANDIT (board 8) reported, send end
                if board_id == 8:
                    time.sleep(0.5)
                    self.pub_score.publish(String(f"{self.team_name},{self.team_pass},-1,NA"))

                return


if __name__ == "__main__":
    try:
        detector = BoardDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
