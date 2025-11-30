#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import message_filters

# Add the script directory to Python path to import model
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from model_vision import PilotNetVision

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/B1/rrbot/camera1/image_raw')
        self.scan_topic = rospy.get_param('~scan_topic', '/B1/scan')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/B1/cmd_vel')
        
        # Default model path
        default_model_path = os.path.join(script_dir, 'best_model_vision_angular.pth')
        self.model_path = rospy.get_param('~model_path', default_model_path)

        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PilotNetVision().to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            rospy.loginfo(f"✓ Loaded model from {self.model_path}")
        else:
            rospy.logwarn(f"✗ Model not found at {self.model_path}")
            rospy.logwarn("  Using random weights (Robot will crash!)")

        self.model.eval()

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        # Track if manual teleop is active
        self.teleop_active = False
        rospy.Subscriber('/teleop_active', Bool, self.teleop_active_callback, queue_size=1)
        
        # Subscribers
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.scan_sub = message_filters.Subscriber(self.scan_topic, LaserScan)
        
        # Sync
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.scan_sub], 
            queue_size=10, 
            slop=0.1,
            allow_headerless=True
        )
        self.ts.registerCallback(self.callback)

        rospy.loginfo("Inference Node Started (Vision Only)")
        
<<<<<<< HEAD
        # Smoothing
        self.smoothing_alpha = 0.5
        self.prev_v = 0.0
        self.prev_w = 0.0
=======
        # Counter for logging
        self.callback_count = 0
>>>>>>> 7720b412679ec3c9c1f7c7453696a796b833d560
    
    def teleop_active_callback(self, msg):
        self.teleop_active = msg.data

    def callback(self, image_msg, scan_msg):
        if self.teleop_active:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Preprocessing Image
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (120, 120))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Watchdog (LIDAR)
        # Check front cone for obstacles
        scan_ranges = np.array(scan_msg.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, posinf=10.0)
        
        # Assuming 720 samples, front is ~360. Check +/- 30 samples (~15 deg)
        center_idx = len(scan_ranges) // 2
        front_cone = scan_ranges[center_idx-30 : center_idx+30]
        
        if len(front_cone) > 0 and np.min(front_cone) < 0.2:
            rospy.logwarn_throttle(1.0, f"Obstacle detected! Stopping. (Dist: {np.min(front_cone):.2f}m)")
            twist = Twist()
            self.pub.publish(twist)
            return

        # Inference - direct output like manual controller
        with torch.no_grad():
            output = self.model(image)
            
            # Denormalize
            # w: [-1, 1] -> [-3, 3]
            w_raw = np.clip(output[0][0].item(), -1, 1) * 3.0
            
            # Smoothing
            w = self.smoothing_alpha * w_raw + (1 - self.smoothing_alpha) * self.prev_w
            self.prev_w = w
            
            # Manual Speed Control Logic
            # "Half it when there is a tight turn"
            # Base speed: 0.5 m/s (safe) or 1.0 m/s (fast)
            # Threshold for "tight turn": 1.0 rad/s
            
            base_speed = 0.5  # Adjust this for competition speed
            turn_threshold = 0.8
            
            if abs(w) > turn_threshold:
                v = base_speed * 0.5  # Half speed for turns
            else:
                v = base_speed
                
            self.prev_v = v

        # Publish directly - no smoothing, just like controller
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub.publish(twist)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = InferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
