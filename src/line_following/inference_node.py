#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import message_filters

# Add the script directory to Python path to import model
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from model import PilotNet

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node', anonymous=True)

        # Parameters - Fixed topic names to match robot namespacing
        self.image_topic = rospy.get_param('~image_topic', '/B1/rrbot/camera1/image_raw')
        self.scan_topic = rospy.get_param('~scan_topic', '/B1/scan')
        self.front_scan_topic = rospy.get_param('~front_scan_topic', '/B1/front_scan')
        self.left_scan_topic = rospy.get_param('~left_scan_topic', '/B1/left_scan')
        self.right_scan_topic = rospy.get_param('~right_scan_topic', '/B1/right_scan')
        self.side_left_scan_topic = rospy.get_param('~side_left_scan_topic', '/B1/side_left_scan')
        self.side_right_scan_topic = rospy.get_param('~side_right_scan_topic', '/B1/side_right_scan')
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
        
        # Track if manual teleop is active
        self.teleop_active = False
        self.last_pub_check_time = rospy.Time.now()
        self.pub_check_interval = 3.0  # Check every 3 seconds
        
        # Track if competition timer has started
        self.timer_started = False
        rospy.Subscriber('/score_tracker', String, self.score_tracker_callback, queue_size=1)
        
        # Counter for logging - must be initialized before callbacks are registered
        self.callback_count = 0
        
        rospy.Subscriber('/teleop_active', Bool, self.teleop_active_callback, queue_size=1)
        
        # Front lidar for obstacle detection (30 degree cone, 0.1m threshold)
        self.front_scan_data = None
        self.obstacle_detected = False
        self.front_obstacle_pause_until = rospy.Time.now()  # Time until which to stay paused after obstacle clears
        rospy.Subscriber(self.front_scan_topic, LaserScan, self.front_scan_callback, queue_size=1)
        
        # Left and Right lidars for obstacle detection
        self.left_obstacle_detected = False
        self.right_obstacle_detected = False
        self.left_hard_stop = False   # Hard stop if object at 0.075m
        self.right_hard_stop = False  # Hard stop if object at 0.075m
        self.left_hard_stop_pause_until = rospy.Time.now()   # Pause timer after left hard stop clears
        self.right_hard_stop_pause_until = rospy.Time.now()  # Pause timer after right hard stop clears
        self.side_left_obstacle_detected = False
        self.side_right_obstacle_detected = False
        rospy.Subscriber(self.left_scan_topic, LaserScan, self.left_scan_callback, queue_size=1)
        rospy.Subscriber(self.right_scan_topic, LaserScan, self.right_scan_callback, queue_size=1)
        rospy.Subscriber(self.side_left_scan_topic, LaserScan, self.side_left_scan_callback, queue_size=1)
        rospy.Subscriber(self.side_right_scan_topic, LaserScan, self.side_right_scan_callback, queue_size=1)
        
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
        rospy.loginfo(f"  Front Scan: {self.front_scan_topic}")
        rospy.loginfo(f"  Left Scan: {self.left_scan_topic}")
        rospy.loginfo(f"  Right Scan: {self.right_scan_topic}")
        rospy.loginfo("Note: Will pause when manual teleop is active to avoid conflicts")
        rospy.loginfo("Note: Waiting for competition timer to start before moving...")
    
    def score_tracker_callback(self, msg):
        """Track when competition timer starts (location '0' message)"""
        if not self.timer_started:
            # Parse the message: "teamName,password,location,clue"
            parts = msg.data.split(',')
            if len(parts) >= 3 and parts[2] == '0':
                self.timer_started = True
                rospy.loginfo("🏁 Competition timer STARTED - Line following enabled!")
    
    def teleop_active_callback(self, msg):
        """Track when manual teleop controller is active"""
        if msg.data != self.teleop_active:
            self.teleop_active = msg.data
            status = "ACTIVE" if self.teleop_active else "INACTIVE"
            rospy.loginfo(f"Manual teleop is {status} - Inference {'PAUSED' if msg.data else 'RESUMED'}")
    
    def front_scan_callback(self, msg):
        """Process front lidar scan for obstacle detection in 25 degree cone"""
        self.front_scan_data = msg
        
        # Get valid ranges (filter out inf and nan, and values outside sensor range)
        ranges = list(msg.ranges)
        valid_ranges = [r for r in ranges if r > msg.range_min and r < msg.range_max and not np.isnan(r) and not np.isinf(r)]
        
        was_obstacle_detected = self.obstacle_detected
        
        if valid_ranges:
            min_dist = min(valid_ranges)
            # Detect obstacle within 0.175m (17.5cm) in the front cone - STOP robot
            self.obstacle_detected = min_dist < 0.175
            
            # Debug logging (throttled)
            rospy.loginfo_throttle(2.0, f"Front LIDAR: min={min_dist:.3f}m, valid={len(valid_ranges)}/{len(ranges)}, obstacle={self.obstacle_detected}")
            
            if self.obstacle_detected:
                rospy.logwarn_throttle(0.5, f"⚠️ FRONT OBSTACLE at {min_dist:.3f}m - STOPPING!")
            elif was_obstacle_detected and not self.obstacle_detected:
                # Obstacle just cleared - set pause timer for 0.5 seconds
                self.front_obstacle_pause_until = rospy.Time.now() + rospy.Duration(0.5)
                rospy.loginfo("Front obstacle cleared - pausing for 0.5s before resuming")
        else:
            if self.obstacle_detected:
                # Was detecting obstacle, now no valid ranges - set pause timer
                self.front_obstacle_pause_until = rospy.Time.now() + rospy.Duration(0.5)
            self.obstacle_detected = False
            rospy.logwarn_throttle(5.0, f"Front LIDAR: No valid ranges (all inf/nan or outside bounds)")
    
    def left_scan_callback(self, msg):
        """Process left lidar scan - steer right if obstacle detected, hard stop at 0.075m"""
        ranges = [r for r in msg.ranges if r > msg.range_min and r < msg.range_max]
        was_hard_stop = self.left_hard_stop
        if ranges:
            min_dist = min(ranges)
            # Hard stop if obstacle within 0.075m (7.5cm)
            if min_dist < 0.075:
                self.left_hard_stop = True
                self.left_obstacle_detected = True
                rospy.logwarn_throttle(0.5, f"🛑 LEFT HARD STOP at {min_dist:.3f}m!")
            elif min_dist < 0.20:
                # Was hard stop, now cleared - set pause timer
                if was_hard_stop:
                    self.left_hard_stop_pause_until = rospy.Time.now() + rospy.Duration(0.5)
                self.left_hard_stop = False
                self.left_obstacle_detected = True
                rospy.logwarn_throttle(0.5, f"⚠️ LEFT FRONT OBSTACLE at {min_dist:.3f}m! Steering right.")
            else:
                # Was hard stop, now cleared - set pause timer
                if was_hard_stop:
                    self.left_hard_stop_pause_until = rospy.Time.now() + rospy.Duration(0.5)
                self.left_hard_stop = False
                self.left_obstacle_detected = False
        else:
            if was_hard_stop:
                self.left_hard_stop_pause_until = rospy.Time.now() + rospy.Duration(0.5)
            self.left_hard_stop = False
            self.left_obstacle_detected = False

    def right_scan_callback(self, msg):
        """Process right lidar scan - steer left if obstacle detected, hard stop at 0.075m"""
        ranges = [r for r in msg.ranges if r > msg.range_min and r < msg.range_max]
        was_hard_stop = self.right_hard_stop
        if ranges:
            min_dist = min(ranges)
            # Hard stop if obstacle within 0.075m (7.5cm)
            if min_dist < 0.075:
                self.right_hard_stop = True
                self.right_obstacle_detected = True
                rospy.logwarn_throttle(0.5, f"🛑 RIGHT HARD STOP at {min_dist:.3f}m!")
            elif min_dist < 0.20:
                # Was hard stop, now cleared - set pause timer
                if was_hard_stop:
                    self.right_hard_stop_pause_until = rospy.Time.now() + rospy.Duration(0.5)
                self.right_hard_stop = False
                self.right_obstacle_detected = True
                rospy.logwarn_throttle(0.5, f"⚠️ RIGHT FRONT OBSTACLE at {min_dist:.3f}m! Steering left.")
            else:
                # Was hard stop, now cleared - set pause timer
                if was_hard_stop:
                    self.right_hard_stop_pause_until = rospy.Time.now() + rospy.Duration(0.5)
                self.right_hard_stop = False
                self.right_obstacle_detected = False
        else:
            if was_hard_stop:
                self.right_hard_stop_pause_until = rospy.Time.now() + rospy.Duration(0.5)
            self.right_hard_stop = False
            self.right_obstacle_detected = False

    def side_left_scan_callback(self, msg):
        """Process side left lidar scan - steer right if obstacle detected"""
        ranges = [r for r in msg.ranges if r > msg.range_min and r < msg.range_max and not np.isnan(r) and not np.isinf(r)]
        if ranges and min(ranges) < 0.15:
            self.side_left_obstacle_detected = True
            rospy.logwarn_throttle(0.5, f"⚠️ SIDE LEFT OBSTACLE at {min(ranges):.3f}m! Steering right.")
        else:
            self.side_left_obstacle_detected = False

    def side_right_scan_callback(self, msg):
        """Process side right lidar scan - steer left if obstacle detected"""
        ranges = [r for r in msg.ranges if r > msg.range_min and r < msg.range_max and not np.isnan(r) and not np.isinf(r)]
        if ranges and min(ranges) < 0.15:
            self.side_right_obstacle_detected = True
            rospy.logwarn_throttle(0.5, f"⚠️ SIDE RIGHT OBSTACLE at {min(ranges):.3f}m! Steering left.")
        else:
            self.side_right_obstacle_detected = False

    def check_publisher_connection(self):
        """Check if publisher is connected and recreate if needed (for sim resets)"""
        current_time = rospy.Time.now()
        if (current_time - self.last_pub_check_time).to_sec() < self.pub_check_interval:
            return
        
        self.last_pub_check_time = current_time
        
        # Check if we have subscribers
        num_connections = self.pub.get_num_connections()
        
        if num_connections == 0 and not self.teleop_active:
            rospy.logwarn_throttle(15.0, 
                f"No subscribers to {self.cmd_vel_topic}. Simulation may have reset. Recreating publisher...")
            self.pub.unregister()
            self.pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
            rospy.sleep(0.1)

    def callback(self, image_msg, scan_msg):
        # Check publisher connection periodically
        self.check_publisher_connection()
        
        # Don't move until competition timer has started
        if not self.timer_started:
            rospy.loginfo_throttle(5.0, "⏳ Waiting for competition timer to start...")
            return
        
        # Don't interfere if manual teleop is active
        if self.teleop_active:
            return
        
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

        # Left/Right front lidar hard stop - emergency stop at 0.075m or in pause period
        left_paused = rospy.Time.now() < self.left_hard_stop_pause_until
        right_paused = rospy.Time.now() < self.right_hard_stop_pause_until
        if self.left_hard_stop or self.right_hard_stop or left_paused or right_paused:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)
            if self.left_hard_stop or self.right_hard_stop:
                side = "LEFT" if self.left_hard_stop else "RIGHT"
                rospy.logwarn_throttle(0.5, f"🛑 HARD STOP - {side} obstacle too close!")
            else:
                side = "LEFT" if left_paused else "RIGHT"
                rospy.loginfo_throttle(0.25, f"⏸️ PAUSED - Waiting after {side} hard stop cleared")
            return

        # Front lidar obstacle detection - stop if obstacle detected or in pause period
        if self.obstacle_detected or rospy.Time.now() < self.front_obstacle_pause_until:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.pub.publish(twist)
            if self.obstacle_detected:
                rospy.logwarn_throttle(0.5, f"🛑 STOP COMMAND SENT - Obstacle in front!")
            else:
                rospy.loginfo_throttle(0.25, f"⏸️ PAUSED - Waiting after obstacle cleared")
            return

        # Inference
        with torch.no_grad():
            output = self.model(image, scan_tensor)
            v = 1.7
            w = 1.44 * output[0][1].item()

        # If max turning speed > 2.0 rad/s, set max velocity to 0.85 m/s
        if abs(w) > 3.0:
            v = min(v, 0.95)

        # Front corner lidar obstacle avoidance - steer away from obstacles
        if self.left_obstacle_detected:
            w -= 0.55  # Steer right (negative angular velocity)
        if self.right_obstacle_detected:
            w += 0.55  # Steer left (positive angular velocity)

        # Side lidar obstacle avoidance - steer away from obstacles
        if self.side_left_obstacle_detected:
            w -= 0.75  # Steer right (negative angular velocity)
            
        if self.side_right_obstacle_detected:
            w += 0.75  # Steer left (positive angular velocity)
            

        twist = Twist()
        # twist.linear.x = v
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
