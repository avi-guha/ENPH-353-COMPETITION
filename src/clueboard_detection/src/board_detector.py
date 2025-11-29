#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import time

class BoardDetector:
    LOWER_BLUE_HSV = np.array([100, 50, 20])
    UPPER_BLUE_HSV = np.array([130, 255, 150])

    # Constructor
    def __init__(self):
        self.bridge = CvBridge()

        curr_time = time()
        last_board_time = time()

        # Hashmap / dict containing all labels of clueboards in course with corresponding clueboard ID (integer).
        # Bool (item 1 in list of value) marks if this clueboard has been reported to /score_tracker yet.
        self.board_map = {
            "SIZE": [1, False],
            "VICTIM": [2, False],
            "CRIME": [3, False],
            "TIME": [4, False],
            "PLACE": [5, False],
            "MOTIVE": [6, False],
            "WEAPON": [7, False],
            "BANDIT": [8, False],
        }

        self.team_name = "teamEMAG"
        self.team_pass = "secret3"

        rospy.init_node('yolo_clueboard_node')
        
        model_path = rospy.get_param("~model_path", 
        "/home/fizzer/ENPH-353-COMPETITION/src/clueboard_detection/runs/detect/clueboards_exp12/weights/best.pt")
        self.model = YOLO(model_path)

        self.camera_topic_left = rospy.get_param("~camera_topic_left", "/B1/left_front_cam/left_front/image_raw")
        self.camera_topic_right = rospy.get_param("~camera_topic_right", "/B1/right_front_cam/right_front/image_raw")

        rospy.Subscriber(self.camera_topic_left, Image, self.camera_callback, queue_size=1)
        rospy.Subscriber(self.camera_topic_right, Image, self.camera_callback, queue_size=1)

        self.pub = rospy.Publisher('/yolo_clueboard/image', Image, queue_size=1)

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

    # Desired crop extraction
    def process_raw_board(self, raw_board):
        """
        @brief Crop YOLO bounded board to grey section.
        @param raw_board the bounded image return from YOLO reference.
        @return Cropped version of raw_board to grey section only.
        """
        return 1
    
    def camera_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLO inference
        results = self.model.predict(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()

                if confidence > 0.91 and bbz_size:
                    frame_extract = frame[y1:y2, x1:x2]
                    # use BGR until cnn step
                    # 1. check if whole board captured
                    if(self.board_captured(frame_extract)):
                        rospy.loginfo(">0.9 confidence board, full board found.")
                        annotated = r.plot()
                        self.pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))


                    # 2. if yes, process to gray region

                    # 3. feed into cnn, extract 2 words predictions

                    # 4. read Label of board (Size, Victim, ...)
                    #    if Label in keys(board_map), proceed
                    #    if previous board has marker 'False', ignore
                    
                    # 5. if safe to continue, report clue to /score_tracker -> WITH NO SPACES! , all caps
                    #    before reporting lets just test by rospy.loginfo


                    #CHANGE TO RGB BEFORE GIVING TO CNN
    


if __name__ == "__main__":
    try:
        detector = BoardDetector()  # Initialize your detector
        rospy.loginfo("BoardDetector node is running...")
        rospy.spin()  # Keep Python from exiting until the node is stopped
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down BoardDetector node.")
