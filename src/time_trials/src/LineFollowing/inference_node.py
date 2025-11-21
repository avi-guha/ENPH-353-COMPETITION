#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from model import PilotNet
import os

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/rrbot/camera1/image_raw')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.model_path = rospy.get_param('~model_path', 'best_model.pth')

        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PilotNet().to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            rospy.loginfo(f"Loaded model from {self.model_path}")
        else:
            rospy.logwarn(f"Model not found at {self.model_path}. Using random weights (Robot will crash!).")

        self.model.eval()

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.callback)

        rospy.loginfo("Inference Node Started")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Preprocessing (Must match training)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (200, 66))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(image)
            v = output[0][0].item()
            w = output[0][1].item()

        # Safety Watchdog (Simple version: Stop if v is too high or low?)
        # Real watchdog needs LIDAR/Depth. 
        # Here we just clamp values to safe limits if needed.
        # v = max(min(v, 0.5), -0.5)
        # w = max(min(w, 1.0), -1.0)

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub.publish(twist)

if __name__ == '__main__':
    try:
        node = InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
