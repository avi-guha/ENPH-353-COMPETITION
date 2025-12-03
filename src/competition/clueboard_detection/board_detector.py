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

        # Timing
        self.last_board_time = time.time()
        self.last_callback_time = 0.0
        self.last_yolo_time = 0.0

        # CPU throttle
        self.callback_interval = 0.05   # process at most ~20 Hz
        self.yolo_interval = 0.8        # YOLO at most ~1.25 Hz

        self.current_board = 0

        # Board sequence tracking
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

        rospy.init_node('yolo_clueboard_node')

        # Publishers
        self.pub_raw_board = rospy.Publisher('/clueboard/raw_board', Image, queue_size=1)
        self.pub_proc_board = rospy.Publisher('/clueboard/processed_board', Image, queue_size=1)
        self.pub_words_debug = rospy.Publisher('/clueboard/words_debug', Image, queue_size=1)
        self.pub_letters_debug = rospy.Publisher('/clueboard/letters_debug', Image, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        # NEW: YOLO visualization publisher
        self.pub_yolo_vis = rospy.Publisher('/yolo_clueboard/image', Image, queue_size=1)

        # Connect BoardReader
        self.cnn.bridge = self.bridge
        self.cnn.pub_words_debug = self.pub_words_debug
        self.cnn.pub_letters_debug = self.pub_letters_debug

        model_path = rospy.get_param(
            "~model_path",
            "/home/fizzer/ENPH-353-COMPETITION/src/competition/clueboard_detection/runs/detect/clueboards_exp12/weights/best.pt"
        )

        self.model = YOLO(model_path)
        # keep your no-op fuse
        self.model.fuse = lambda *a, **k: self.model

        self.camera_topic_left = rospy.get_param(
            "~camera_topic_left",
            "/B1/left_front_cam/left_front/image_raw"
        )
        self.camera_topic_right = rospy.get_param(
            "~camera_topic_right",
            "/B1/right_front_cam/right_front/image_raw"
        )

        rospy.Subscriber(self.camera_topic_left, Image, self.camera_callback_left, queue_size=1)
        rospy.Subscriber(self.camera_topic_right, Image, self.camera_callback_right, queue_size=1)

        rospy.loginfo("Board Detector Initialized (with YOLO visualization).")

    # ------------------ CPU helpers ------------------
    def should_process_callback(self):
        t = time.time()
        if t - self.last_callback_time < self.callback_interval:
            return False
        self.last_callback_time = t
        return True

    def should_run_yolo(self):
        t = time.time()
        if t - self.last_yolo_time < self.yolo_interval:
            return False
        self.last_yolo_time = t
        return True

    # ---------------- YOLO Visualization ----------------
    def publish_yolo_vis(self, frame, results):
        vis = frame.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = r.names[cls] if hasattr(r, "names") else f"id:{cls}"

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(vis,
                            f"{label} {conf:.2f}",
                            (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

        self.pub_yolo_vis.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

    # ---------------- Board validation ----------------
    def board_captured(self, raw_board):
        hsv_image = cv2.cvtColor(raw_board, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.LOWER_BLUE_HSV, self.UPPER_BLUE_HSV)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
        n = len(approx)

        return (n == 4) or (3 <= n <= 5)

    # ---------------- Crop/warp board ----------------
    def process_raw_board(self, raw_board):
        img = raw_board.copy()
        H, W = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (90, 60, 20), (130, 255, 255))
        cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img
        outer = max(cnts, key=cv2.contourArea)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11,11), 0)
        edges = cv2.Canny(blur, 10, 40)

        mask = np.zeros((H,W), np.uint8)
        cv2.drawContours(mask, [outer], -1, 255, -1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        edges = cv2.dilate(edges, k, 2)

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
            pts[np.argmax(diff)]
        ], np.float32)

        w1 = np.linalg.norm(pts[0]-pts[1])
        w2 = np.linalg.norm(pts[3]-pts[2])
        h1 = np.linalg.norm(pts[0]-pts[3])
        h2 = np.linalg.norm(pts[1]-pts[2])

        W_out = int(max(w1,w2))
        H_out = int(max(h1,h2))

        dst = np.array([[0,0],[W_out-1,0],[W_out-1,H_out-1],[0,H_out-1]], np.float32)

        M = cv2.getPerspectiveTransform(pts, dst)
        rectified = cv2.warpPerspective(img, M, (W_out,H_out))

        trim = 8
        if rectified.shape[0] > 2*trim and rectified.shape[1] > 2*trim:
            rectified = rectified[trim:-trim, trim:-trim]

        return rectified

    # --------------- Shared left/right camera logic ---------------
    def handle_camera(self, msg, forbidden):
        # START case
        if self.current_board == 0:
            out = String()
            out.data = f"{self.team_name},{self.team_pass},0,NA"
            self.pub_score.publish(out)
            self.current_board = 1
            return

        # DONE case
        if self.current_board > 8:
            out = String()
            out.data = f"{self.team_name},{self.team_pass},-1,NA"
            self.pub_score.publish(out)
            return

        # CPU throttles
        if not self.should_process_callback():
            return

        if self.current_board in forbidden:
            return

        if time.time() - self.last_board_time < 3.0:
            return

        if not self.should_run_yolo():
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        H, W = frame.shape[:2]

        # YOLO FULL-RES
        results = self.model.predict(frame, verbose=False)

        # Publish YOLO visualization to RViz
        self.publish_yolo_vis(frame, results)

        # Interpret YOLO results
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])

                bw = x2 - x1
                bh = y2 - y1

                sizeable = bw > max(120, 0.25*W)
                aspectratio = bw / max(bh, 1) > 1.35
                confident = conf > 0.70

                ordered = (
                    self.current_board-1 in self.board_map and
                    self.board_map[self.current_board-1][0] and
                    not self.board_map[self.current_board][0]
                )

                if not (sizeable and aspectratio and confident and ordered):
                    continue

                frame_extract = frame[y1:y2, x1:x2]

                if not self.board_captured(frame_extract):
                    continue

                rospy.loginfo(f"Board {self.current_board} found at high confidence.")

                roi = self.process_raw_board(frame_extract)
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                self.pub_raw_board.publish(self.bridge.cv2_to_imgmsg(frame_extract, "bgr8"))
                self.pub_proc_board.publish(self.bridge.cv2_to_imgmsg(roi, "bgr8"))

                pred_chars = self.cnn.predict_board(rgb_roi)
                words = "".join(pred_chars).split()
                if not words:
                    return

                predicted_label = words[0]
                expected_label = self.board_map[self.current_board][1]

                if predicted_label != expected_label:
                    rospy.logwarn(f"Label mismatch: CNN '{predicted_label}' != expected '{expected_label}'")
                    return

                clue = "".join(w.replace(" ", "") for w in words[1:])
                out = String()
                out.data = f"{self.team_name},{self.team_pass},{self.current_board},{clue}"
                self.pub_score.publish(out)

                self.board_map[self.current_board][0] = True
                self.last_board_time = time.time()
                self.current_board += 1

                return

    def camera_callback_left(self, msg):
        self.handle_camera(msg, forbidden=[2,4,6,8])

    def camera_callback_right(self, msg):
        self.handle_camera(msg, forbidden=[1,3,5,7])


if __name__ == "__main__":
    try:
        BoardDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
