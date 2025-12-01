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

    # Constructor
    def __init__(self):
        self.cnn = BoardReader()

        self.bridge = CvBridge()

        self.curr_time = time.time()
        self.last_board_time = time.time()

        self.current_board = 1

        # Hashmap / dict containing all IDs of clueboards in course with corresponding bool, and Label.
        # Bool (item 0 in list of value) marks if this clueboard has been reported to /score_tracker yet.
        self.board_map = {
            0: [True, "START"], #assume LF has begun already
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

        # init publishers for GUI
        self.pub_raw_board = rospy.Publisher('/clueboard/raw_board', Image, queue_size=1)
        self.pub_proc_board = rospy.Publisher('/clueboard/processed_board', Image, queue_size=1)
        self.pub_words_debug = rospy.Publisher('/clueboard/words_debug', Image, queue_size=1)
        self.pub_letters_debug = rospy.Publisher('/clueboard/letters_debug', Image, queue_size=1)

        # publishers for debug images
        self.pub_words_debug = rospy.Publisher('/clueboard/words_debug', Image, queue_size=1)
        self.pub_letters_debug = rospy.Publisher('/clueboard/letters_debug', Image, queue_size=1)

        # Give BoardReader access to publishers
        self.cnn.bridge = self.bridge
        self.cnn.pub_words_debug = self.pub_words_debug
        self.cnn.pub_letters_debug = self.pub_letters_debug
        
        model_path = rospy.get_param("~model_path", 
        "/home/fizzer/ENPH-353-COMPETITION/src/competition/clueboard_detection/runs/detect/clueboards_exp12/weights/best.pt")
        self.model = YOLO(model_path)
        self.model.fuse = lambda *args, **kwargs: self.model

        self.camera_topic_left = rospy.get_param("~camera_topic_left", "/B1/left_front_cam/left_front/image_raw")
        self.camera_topic_right = rospy.get_param("~camera_topic_right", "/B1/right_front_cam/right_front/image_raw")

        rospy.Subscriber(self.camera_topic_left, Image, self.camera_callback, queue_size=1)
        rospy.Subscriber(self.camera_topic_right, Image, self.camera_callback, queue_size=1)

        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)

        self.pub_yolo = rospy.Publisher('/yolo_clueboard/image', Image, queue_size=1)

        rospy.loginfo("Board Detector Initialized!")
    
    # Board validation
    def board_captured(self, raw_board):
        """
        @brief Determine if the entire clueboard is captured in image.
        @param raw_board the bounded image return from YOLO reference.
        @return True if the whole board is captured in the image; false if only a segment of it is captured.
        """
        # Isolate blue border using HSV 
        hsv_image = cv2.cvtColor(raw_board, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_image, self.LOWER_BLUE_HSV, self.UPPER_BLUE_HSV)

        # Find the largest continuous blue region (the border itself)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # The epsilon parameter defines the maximum distance between the original contour and the approximated polygon
        epsilon = 0.04 * perimeter 
        
        # Approximate the contour with fewer vertices
        approx_poly = cv2.approxPolyDP(largest_contour, epsilon, True)
        num_corners = len(approx_poly)
        
        # If it has exactly 4 corners, it is a perfect quadrilateral.
        if num_corners == 4:
            return True
        
        # You can adjust this range based on testing. 4 is the most definitive value.
        return 3 <= num_corners <= 5

    def process_raw_board(self, raw_board):
        """
        Geometric homography-based solution.
        Extracts the INNER GREY QUAD (not the blue border).
        No thresholds. No band cropping. No empty outputs.
        Produces a perfect, borderless rectangular board interior.
        """
        img = raw_board.copy()
        H, W = img.shape[:2]

        # Detect blue border
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (90,60,20), (130,255,255))

        cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img  # fallback

        outer = max(cnts, key=cv2.contourArea)

        # Create mask inside blue border
        border_mask = np.zeros((H,W), np.uint8)
        cv2.drawContours(border_mask, [outer], -1, 255, -1)

        # Blur to remove letter strokes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11,11), 0)

        # edges inside the border only
        edges = cv2.Canny(blur, 10, 40)
        edges = cv2.bitwise_and(edges, edges, mask=border_mask)

        # Dilate to close grey-panel perimeter
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        edges = cv2.dilate(edges, k, iterations=2)

        cnts2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts2:
            return img

        inner = max(cnts2, key=cv2.contourArea)

        # Approx inner border to 4 points
        peri = cv2.arcLength(inner, True)
        approx = cv2.approxPolyDP(inner, 0.03 * peri, True)

        if len(approx) != 4:
            # fallback: if weird contour, crop boundingRect
            x,y,w,h = cv2.boundingRect(inner)
            return img[y:y+h, x:x+w]

        pts_src = approx.reshape(4,2)

        # order the points TL, TR, BR, BL
        def order_pts(pts):
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            return np.array([
                pts[np.argmin(s)],      # TL
                pts[np.argmin(diff)],   # TR
                pts[np.argmax(s)],      # BR
                pts[np.argmax(diff)]    # BL
            ])

        pts_src = order_pts(pts_src).astype(np.float32)

        # Define output size
        # estimate width / height from distances
        w1 = np.linalg.norm(pts_src[0] - pts_src[1])
        w2 = np.linalg.norm(pts_src[3] - pts_src[2])
        h1 = np.linalg.norm(pts_src[0] - pts_src[3])
        h2 = np.linalg.norm(pts_src[1] - pts_src[2])

        W_out = int(max(w1, w2))
        H_out = int(max(h1, h2))

        pts_dst = np.array([
            [0, 0],
            [W_out-1, 0],
            [W_out-1, H_out-1],
            [0, H_out-1]
        ], dtype=np.float32)

        # Homography
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        rectified = cv2.warpPerspective(img, M, (W_out, H_out))
        
        # Trim pixels from edges
        trim = 8 

        Hf, Wf = rectified.shape[:2]

        if Hf > 2*trim and Wf > 2*trim:
            rectified = rectified[trim:Hf-trim, trim:Wf-trim]

        return rectified


    def camera_callback(self, msg):
        """
        @brief Callback function on L/R camera topics, triggered when new image received. Handles board detection logic.
        @param msg the callback Image
        """
        # early exit if current_board is out of range
        if self.current_board not in self.board_map:
            return

        # enforce cooldown BEFORE running YOLO to prevent double triggers
        if (time.time() - self.last_board_time) < 3.0: 
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLO inference
        results = self.model.predict(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()

                # ensure all conditions met before proceeding to read board
                sizeable  = (x2-x1) > 450
                aspectratio = (x2-x1) / (y2-y1) > 1.4
                confident = confidence > 0.91

                # validate order of board detection, ensure this baord isnt seen yet
                ordered = (
                    self.current_board-1 in self.board_map and
                    self.board_map[self.current_board-1][0] and 
                    not self.board_map[self.current_board][0]
                )

                if sizeable and aspectratio and confident and ordered:
                    frame_extract = frame[y1:y2, x1:x2]

                    # check if whole board captured
                    if(self.board_captured(frame_extract)):
                        rospy.loginfo(">0.9 confidence and full board found.")

                        # process to gray region (you already planned this)
                        roi = self.process_raw_board(frame_extract)

                        # convert to rgb for cnn
                        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                        # publish for GUI
                        self.pub_raw_board.publish(self.bridge.cv2_to_imgmsg(frame_extract, "bgr8"))
                        self.pub_proc_board.publish(self.bridge.cv2_to_imgmsg(roi, "bgr8"))

                        # feed into cnn, extract predicted LABEL for board                        
                        pred_chars = self.cnn.predict_board(rgb_roi)
                        words = "".join(pred_chars).split()

                        predicted_label = words[0]

                        # validate that expected label matches CNN label  
                        expected_label = self.board_map[self.current_board][1]

                        if predicted_label != expected_label:
                            rospy.logwarn(f"Ignoring board: CNN label '{predicted_label}' "
                                          f"does not match expected '{expected_label}'")
                            return
                                    
                        #only read rest of board if passed check
                        predicted_text = [w.replace(" ", "") for w in words[1:]]

                        # debug
                        """
                        rospy.loginfo(f"---BOARD FOUND! ID: {self.current_board}, LABEL: {predicted_label}")
                        rospy.loginfo(f"---BOARD TEXT: {predicted_text}")
                        rospy.loginfo("------------------------------------------")
                        """

                        # report clue to /score_tracker -> WITH NO SPACES! 
                        clue_text = "".join(predicted_text)

                        msg_out = String()
                        msg_out.data = f"{self.team_name},{self.team_pass},{self.current_board},{clue_text}"

                        self.pub_score.publish(msg_out)

                        
                        # mark board as reported
                        self.board_map[self.current_board][0] = True
                        self.last_board_time = time.time()

                        # --- CHANGE: increment safely, prevent overflow
                        if self.current_board < max(self.board_map.keys()):
                            self.current_board += 1
                        
                        # return immediately to avoid multiple detections per callback
                        return


if __name__ == "__main__":
    try:
        detector = BoardDetector()  # Initialize your detector
        rospy.loginfo("BoardDetector node is running...")
        rospy.spin()  # Keep Python from exiting until the node is stopped
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down BoardDetector node.")