#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

def main():

    rospy.init_node('yolo_clueboard_node')

    # Load parameters (optional)
    model_path = rospy.get_param("~model_path", 
        "/home/fizzer/ENPH-353-COMPETITION/src/clueboard_detection/runs/detect/clueboards_exp1/weights/best.pt")
    camera_topic = rospy.get_param("~camera_topic", "/B1/rrbot/camera1/image_raw")

    # Load YOLO model
    model = YOLO(model_path)

    bridge = CvBridge()
    pub = rospy.Publisher('/yolo_clueboard/image', Image, queue_size=1)

    rospy.loginfo(f"[YOLO] Subscribing to camera: {camera_topic}")
    rospy.loginfo(f"[YOLO] Loaded model: {model_path}")

    def callback(msg):
        # Convert ROS → OpenCV
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLO
        results = model(frame)

        # Get annotated frame (WITH bounding boxes)
        annotated = results[0].plot()

        # Publish back to ROS
        pub.publish(bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))

    rospy.Subscriber(camera_topic, Image, callback, queue_size=1)

    rospy.loginfo("YOLO clueboard detection node running...")
    rospy.spin()


if __name__ == "__main__":
    main()
