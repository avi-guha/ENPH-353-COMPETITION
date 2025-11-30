#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import message_filters

from model_angular import PilotNet_AngularOnly

class LineFollowerNode:
    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load model
        self.model = PilotNet_AngularOnly().to(self.device)
        model_path = rospy.get_param('~model_path', '/home/fizzer/ros_ws/src/time_trials/src/LineFollowing/model_angular.pth')
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            rospy.loginfo(f"Model loaded from {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            rospy.signal_shutdown("Model loading failed")
            return
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        
        # Subscribers using message_filters for synchronization
        image_sub = message_filters.Subscriber('/R1/pi_camera/image_raw', Image)
        scan_sub = message_filters.Subscriber('/R1/scan', LaserScan)
        
        # Time synchronizer
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, scan_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)
        
        # Teleop active flag
        self.teleop_active = False
        rospy.Subscriber('/teleop_active', Bool, self.teleop_active_callback)
        
        # Counter for logging
        self.callback_count = 0
        
        # Velocity control parameters
        self.BASE_VELOCITY = 0.5  # Base linear velocity (m/s)
        self.MAX_VELOCITY = 1.0   # Maximum velocity on straights
        self.MIN_VELOCITY = 0.3   # Minimum velocity on sharp turns
        
        rospy.loginfo("Line Follower Node Initialized")
    
    def teleop_active_callback(self, msg):
        """Track when manual teleop controller is active"""
        self.teleop_active = msg.data
        if self.teleop_active:
            rospy.loginfo("Teleop active - inference paused")
        else:
            rospy.loginfo("Teleop inactive - inference resumed")
    
    def callback(self, image_msg, scan_msg):
        # Skip if teleop is active
        if self.teleop_active:
            return
        
        self.callback_count += 1
        
        # Log first few callbacks
        if self.callback_count <= 3:
            rospy.loginfo(f"Callback triggered (count: {self.callback_count})")
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Preprocess Image
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (120, 120))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Preprocess Scan
        scan_ranges = list(scan_msg.ranges)
        scan_ranges = [30.0 if x == float('inf') else x for x in scan_ranges]
        scan_tensor = torch.tensor(scan_ranges, dtype=torch.float32).unsqueeze(0).to(self.device)
        scan_tensor = scan_tensor / 30.0

        # Watchdog Check - front cone obstacle detection
        front_cone = scan_ranges[300:420]  # +/- 15 degrees
        min_dist = min(front_cone)
        
        if min_dist < 0.15:
            rospy.logwarn(f"Obstacle at {min_dist:.2f}m! Stopping.")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)
            return

        # Inference - predict angular velocity only
        with torch.no_grad():
            output = self.model(image, scan_tensor)
            
            # Model outputs Tanh-bounded value in [-1, 1]
            # Denormalize to angular velocity range [-3, 3] rad/s
            w_normalized = output[0][0].item()  # Already in [-1, 1]
            w = w_normalized * 3.0

        # Adaptive velocity control based on steering angle
        # Slower on sharp turns, faster on straights
        abs_w = abs(w)
        if abs_w < 0.5:  # Nearly straight
            v = self.MAX_VELOCITY
        elif abs_w < 1.5:  # Moderate turn
            v = self.BASE_VELOCITY
        else:  # Sharp turn
            v = self.MIN_VELOCITY

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        
        self.pub.publish(twist)
        
        # Log occasionally
        if self.callback_count % 50 == 0:
            rospy.loginfo(f"Predicted: v={v:.2f} m/s, w={w:.2f} rad/s ({w*57.3:.1f}°/s)")

if __name__ == '__main__':
    try:
        node = LineFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
