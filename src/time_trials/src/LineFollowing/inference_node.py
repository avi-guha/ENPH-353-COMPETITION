#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from model import PilotNet
import os
import message_filters

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node', anonymous=True)

        # Parameters - Fixed topic names to match robot namespacing
        self.image_topic = rospy.get_param('~image_topic', '/B1/rrbot/camera1/image_raw')
        self.scan_topic = rospy.get_param('~scan_topic', '/B1/scan')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/B1/cmd_vel')
        
        # Default model path is in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(script_dir, 'best_model.pth')
        self.model_path = rospy.get_param('~model_path', default_model_path)

        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PilotNet().to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            rospy.loginfo(f"✓ Loaded model from {self.model_path}")
        else:
            rospy.logwarn(f"✗ Model not found at {self.model_path}")
            rospy.logwarn(f"  Checked absolute path: {os.path.abspath(self.model_path)}")
            rospy.logwarn("  Using random weights (Robot will crash!)")

        self.model.eval()

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        # Subscribers
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.scan_sub = message_filters.Subscriber(self.scan_topic, LaserScan)
        
        # Sync - allow_headerless for topics without headers
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.scan_sub], 
            queue_size=10, 
            slop=0.1,
            allow_headerless=True
        )
        self.ts.registerCallback(self.callback)

        rospy.loginfo("Inference Node Started")
        rospy.loginfo(f"Publishing to: {self.cmd_vel_topic}")
        rospy.loginfo(f"Subscribing to:")
        rospy.loginfo(f"  Image: {self.image_topic}")
        rospy.loginfo(f"  Scan: {self.scan_topic}")
        
        # Counter for logging
        self.callback_count = 0

    def callback(self, image_msg, scan_msg):
        self.callback_count += 1
        
        # Log first few callbacks to verify it's working
        if self.callback_count <= 3:
            rospy.loginfo(f"Callback triggered (count: {self.callback_count})")
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Preprocessing Image
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (120, 120))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Preprocessing Scan
        scan_ranges = list(scan_msg.ranges)
        scan_ranges = [30.0 if x == float('inf') else x for x in scan_ranges]
        scan_tensor = torch.tensor(scan_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)
        scan_tensor = scan_tensor / 30.0

        # Watchdog Check
        # Check only the front cone (approx +/- 20 degrees)
        # 720 samples cover 180 degrees (3.14 rad)
        # Center is 360. 20 degrees is approx 1/9 of 180, so 80 samples.
        # Range: 360 - 80 = 280 to 360 + 80 = 440
        front_cone = scan_ranges[280:440]
        min_dist = min(front_cone)
        
        if min_dist < 0.2:
            rospy.logwarn(f"Obstacle detected in front cone at {min_dist:.2f}m! Stopping.")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)
            return

        # Inference
        with torch.no_grad():
            output = self.model(image, scan_tensor)
            v = output[0][0].item()
            w = output[0][1].item()

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub.publish(twist)
        
        # Log occasionally to verify commands
        if self.callback_count % 30 == 0:  # Every 30 callbacks
            rospy.loginfo(f"Publishing cmd_vel: v={v:.3f}, w={w:.3f}")

if __name__ == '__main__':
    try:
        node = InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
