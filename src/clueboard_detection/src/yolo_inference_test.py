#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
from PIL import Image as PilImage  # Rename to avoid conflict
import time

output_path = "/home/fizzer/ENPH-353-COMPETITION/src/clueboard_detection/yolo_inference_images/"

last_time = time.time()  # seconds since the epoch
i = 0

def board_captured(raw_board):
    """
    @brief Determine if the entire clueboard is captured in image.
    @param raw_board the bounded image return from YOLO reference.
    @return True if the whole board is captured in the image; false if only a segment of it is captured.
    """
    board_captured = False

    return board_captured

def process_raw_board(raw_board):
    """
    @brief Crop YOLO bounded board to grey section.
    @param raw_board the bounded image return from YOLO reference.
    @return Cropped version of raw_board to grey section only.
    """
    return crop

def main():
    rospy.init_node('yolo_clueboard_node')

    model_path = rospy.get_param("~model_path", 
        "/home/fizzer/ENPH-353-COMPETITION/src/clueboard_detection/runs/detect/clueboards_exp12/weights/best.pt")
    camera_topic = rospy.get_param("~camera_topic", "/B1/left_front_cam/left_front/image_raw")

    model = YOLO(model_path)

    bridge = CvBridge()
    pub = rospy.Publisher('/yolo_clueboard/image', Image, queue_size=1)

    rospy.loginfo(f"[YOLO] Subscribing to camera: {camera_topic}")
    rospy.loginfo(f"[YOLO] Loaded model: {model_path}")

    def callback(msg):
        global last_time, i

        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = model.predict(frame, verbose=False)

        annotated = results[0].plot()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = box.conf[0].item()
                class_id = box.cls[0].item()

                print(f"Bounding Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {confidence:.2f}")

                current_time = time.time()
                if confidence > 0.9 and (current_time - last_time > 5):

                    frame_extract = frame[y1:y2, x1:x2]

                    #cv2.imwrite(f"{output_path}img_{i}.png", frame_extract)
                    i += 1
                    last_time = current_time  # Update timer

        pub.publish(bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

    rospy.Subscriber(camera_topic, Image, callback, queue_size=1)

    rospy.loginfo("YOLO clueboard detection node running...")
    rospy.spin()


if __name__ == "__main__":
    main()