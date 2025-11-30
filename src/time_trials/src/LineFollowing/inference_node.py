#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
import os
import sys
import message_filters
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

# Add the script directory to Python path to import model
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, 'training')
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from model import MultiModalPolicyNet

# Constants matching training
IMG_H, IMG_W = 120, 120
MAX_LIDAR_DIST = 30.0
MAX_V = 2.5
MAX_W = 3.5

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/B1/rrbot/camera1/image_raw')
        self.scan_topic = rospy.get_param('~scan_topic', '/B1/scan')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/B1/cmd_vel')
        
        # Default model path
        default_model_path = os.path.join(script_dir, 'models/best_model.pth')
        self.model_path = rospy.get_param('~model_path', default_model_path)

        # Load Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        self.model = MultiModalPolicyNet().to(self.device)
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                rospy.loginfo(f"✓ Loaded model from {self.model_path}")
            except Exception as e:
                rospy.logerr(f"Failed to load model: {e}")
        else:
            rospy.logwarn(f"✗ Model not found at {self.model_path}")
            rospy.logwarn("  Using random weights (Robot will crash!)")

        self.model.eval()

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        # Track if manual teleop is active
        self.teleop_active = False
        rospy.Subscriber('/teleop_active', Bool, self.teleop_active_callback, queue_size=1)
        
        # State tracking (Last commanded velocity)
        self.current_v = 0.0
        self.current_w = 0.0
        
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

        rospy.loginfo("Multi-Modal Inference Node Started")
        
        # Smoothing
        self.smoothing_alpha = 0.5
        self.prev_v_cmd = 0.0
        self.prev_w_cmd = 0.0
    
    def teleop_active_callback(self, msg):
        self.teleop_active = msg.data

    def preprocess_image(self, cv_image):
        # Resize
        image = cv2.resize(cv_image, (IMG_W, IMG_H))
        # BGR to RGB (Training used cv2.imread which is BGR, then converted to RGB in dataset)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize [0, 1]
        image = image.astype(np.float32) / 255.0
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        # Add batch dim
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        return image

    def preprocess_lidar(self, scan_msg):
        scan = np.array(scan_msg.ranges, dtype=np.float32)
        # Handle Infs/NaNs
        scan = np.nan_to_num(scan, nan=MAX_LIDAR_DIST, posinf=MAX_LIDAR_DIST, neginf=0.0)
        # Handle 0.0 as max range
        scan[scan == 0] = MAX_LIDAR_DIST
        # Clip
        scan = np.clip(scan, 0, MAX_LIDAR_DIST)
        # Resize/Pad to 720 if needed
        if len(scan) != 720:
            if len(scan) > 720:
                scan = scan[:720]
            else:
                scan = np.pad(scan, (0, 720 - len(scan)), 'constant', constant_values=MAX_LIDAR_DIST)
        # Normalize
        scan = scan / MAX_LIDAR_DIST
        # Add batch/channel dims: (B, 1, 720) or just (B, 720) depending on encoder
        # Encoder expects (B, 720) or (B, 1, 720). Let's provide (B, 720)
        scan = torch.tensor(scan, dtype=torch.float32).unsqueeze(0).to(self.device)
        return scan



    def callback(self, image_msg, scan_msg):
        if self.teleop_active:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # 1. Preprocess Inputs
        img_tensor = self.preprocess_image(cv_image)
        lidar_tensor = self.preprocess_lidar(scan_msg)

        # 2. Watchdog Safety Override
        # Check front cone (center +/- 30 degrees approx)
        scan_ranges = np.array(scan_msg.ranges)
        scan_ranges = np.nan_to_num(scan_ranges, posinf=10.0)
        center_idx = len(scan_ranges) // 2
        # Assuming 720 points over ~270 degrees? or 360? 
        # If 720 points, +/- 40 points is a reasonable cone
        front_cone = scan_ranges[center_idx-40 : center_idx+40]
        
        if len(front_cone) > 0 and np.min(front_cone) < 0.3: # 30cm stop distance
            rospy.logwarn_throttle(1.0, f"Obstacle detected! Stopping. (Dist: {np.min(front_cone):.2f}m)")
            twist = Twist()
            self.pub.publish(twist)
            # Reset state
            self.current_v = 0.0
            self.current_w = 0.0
            return

        # 3. Inference
        with torch.no_grad():
            # Output is [v_norm, w_norm]
            output = self.model(img_tensor, lidar_tensor)
            
            v_norm_pred = output[0][0].item()
            w_norm_pred = output[0][1].item()
            
            # Denormalize
            v_cmd = v_norm_pred * MAX_V
            w_cmd = w_norm_pred * MAX_W
            
            # Clip to safe limits
            v_cmd = np.clip(v_cmd, -MAX_V, MAX_V)
            w_cmd = np.clip(w_cmd, -MAX_W, MAX_W)
            
            # Smoothing
            v_cmd = self.smoothing_alpha * v_cmd + (1 - self.smoothing_alpha) * self.prev_v_cmd
            w_cmd = self.smoothing_alpha * w_cmd + (1 - self.smoothing_alpha) * self.prev_w_cmd
            
            self.prev_v_cmd = v_cmd
            self.prev_w_cmd = w_cmd
            
            # Update state for next iteration
            self.current_v = v_cmd
            self.current_w = w_cmd

        # 4. Publish
        twist = Twist()
        twist.linear.x = v_cmd
        twist.angular.z = w_cmd
        self.pub.publish(twist)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = InferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
