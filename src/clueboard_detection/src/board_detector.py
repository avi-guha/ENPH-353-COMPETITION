#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import time
from board_reader import BoardReader

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
    
    def validate_previous(self, board_id):
        """
        @brief Ensure that board at index board_id - 1 has been reported.
        @param board_id the board we are ensuring that previous is accounted for.
        @return True if the baord with id (board_id - 1) has been solved/reported, False otherwise
        """
        previously_done = self.board_map[board_id-1][0]

        if(previously_done):
            return True
        else:
            return False

    def camera_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLO inference
        results = self.model.predict(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()

                # ensure all conditions met before proceeding to read board
                sizeable  = (x2-x1) > 70
                confident = confidence > 0.91
                ordered = self.validate_previous(self.current_board)
                cooloff = (time() - self.last_board_time) > 2

                if sizeable and confident and ordered and cooloff:
                    frame_extract = frame[y1:y2, x1:x2]
                    # use BGR until cnn step
                    # check if whole board captured
                    if(self.board_captured(frame_extract)):
                        rospy.loginfo(">0.9 confidence board, full board found.")
                        #annotated = r.plot()
                        #self.pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

                        # 2. if yes, process to gray region

                        # 3. feed into cnn, extract 2 words predictions  #CHANGE TO RGB BEFORE GIVING TO CNN

                        # 4. read Label of board (Size, Victim, ...)
                        
                        # 5. if safe to continue, report clue to /score_tracker -> WITH NO SPACES! 
                        #    before reporting lets just test by rospy.loginfo

                        self.current_board += 1

    


if __name__ == "__main__":
    try:
        detector = BoardDetector()  # Initialize your detector
        rospy.loginfo("BoardDetector node is running...")
        rospy.spin()  # Keep Python from exiting until the node is stopped
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down BoardDetector node.")
