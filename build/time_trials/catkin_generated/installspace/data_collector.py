#!/usr/bin/env python3

import rospy
import cv2
import os
import csv
from sensor_msgs.msg import Image, LaserScan, Joy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import datetime

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)

        # Parameters - Fixed topic names to match robot namespacing
        self.image_topic = rospy.get_param('~image_topic', '/B1/rrbot/camera1/image_raw')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/B1/cmd_vel')
        self.scan_topic = rospy.get_param('~scan_topic', '/B1/scan')
        
        # Default data directory - in LineFollowing folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_data_dir = os.path.join(script_dir, 'data')
        self.data_dir = rospy.get_param('~data_dir', default_data_dir)

        # Create data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Determine next run number
        run_id = 0
        while os.path.exists(os.path.join(self.data_dir, f'run_{run_id}')):
            run_id += 1
        
        self.run_dir = os.path.join(self.data_dir, f'run_{run_id}')
        os.makedirs(self.run_dir)
        rospy.loginfo(f"Starting new data collection run: {self.run_dir}")

        self.images_dir = os.path.join(self.run_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        self.csv_file_path = os.path.join(self.run_dir, 'log.csv')
        
        # Initialize CSV
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'v', 'w', 'scan'])

        self.bridge = CvBridge()
        self.count = 0
        self.recording = False
        self.last_button_state = 0
        self.button_debounce_time = rospy.Time.now()
        self.debounce_duration = rospy.Duration(0.3)  # 300ms debounce
        
        # Rate limiting to prevent overfitting - 10 fps
        self.target_fps = 10
        self.min_time_between_samples = 1.0 / self.target_fps
        self.last_sample_time = rospy.Time.now()
        
        # Flag to log scan receipt only once
        self.scan_received_logged = False
        
        # Store latest messages
        self.latest_twist = Twist()
        self.latest_scan = None
        self.last_image_time = None
        self.reset_check_time = rospy.Time.now()

        # Subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=1)

        rospy.loginfo(f"Data Collector ready. Saving to {self.data_dir}")
        rospy.loginfo(f"Data collection rate: {self.target_fps} fps")
        rospy.loginfo(f"Subscribed to:")
        rospy.loginfo(f"  Image: {self.image_topic}")
        rospy.loginfo(f"  Cmd_vel: {self.cmd_vel_topic}")
        rospy.loginfo(f"  Scan: {self.scan_topic}")
        rospy.loginfo("Press TRIANGLE (Button 2) to start/stop recording.")
        rospy.loginfo("Recording is currently: PAUSED")

    def joy_callback(self, data):
        """Handle joystick button presses with debouncing"""
        # Triangle button is usually index 2 (check your controller mapping)
        # We want to toggle on press (0 -> 1 transition) with debouncing
        try:
            if len(data.buttons) < 3:
                return
                
            button_state = data.buttons[2]
            current_time = rospy.Time.now()
            
            # Detect button press (0 -> 1 transition) with debounce
            if button_state == 1 and self.last_button_state == 0:
                # Check if enough time has passed since last press
                if (current_time - self.button_debounce_time) > self.debounce_duration:
                    self.recording = not self.recording
                    status = "STARTED" if self.recording else "PAUSED"
                    rospy.loginfo(f"=== Recording {status} ===")
                    self.button_debounce_time = current_time
            
            self.last_button_state = button_state
        except IndexError:
            rospy.logwarn_throttle(10.0, "Controller button mapping error - check your PS4 controller connection")

    def cmd_vel_callback(self, msg):
        self.latest_twist = msg

    def scan_callback(self, msg):
        self.latest_scan = msg
        if not self.scan_received_logged:  # Log only once
            rospy.loginfo(f"✓ Received scan data with {len(msg.ranges)} samples")
            self.scan_received_logged = True
    
    def check_for_reset(self):
        """Check if simulation has reset and reconnect if needed"""
        current_time = rospy.Time.now()
        
        # Only check every 3 seconds
        if (current_time - self.reset_check_time).to_sec() < 3.0:
            return
        
        self.reset_check_time = current_time
        
        # If we had images but haven't received one in 5 seconds, might be a reset
        if self.last_image_time is not None:
            time_since_image = (current_time - self.last_image_time).to_sec()
            if time_since_image > 5.0:
                rospy.logwarn(f"No images for {time_since_image:.1f}s - simulation may have reset. Reconnecting...")
                
                # Recreate subscribers
                self.image_sub.unregister()
                self.scan_sub.unregister()
                self.cmd_vel_sub.unregister()
                
                rospy.sleep(0.2)
                
                self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
                self.cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=1)
                self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
                
                # Reset flags
                self.last_image_time = None
                self.scan_received_logged = False
                rospy.loginfo("Subscribers reconnected")

    def image_callback(self, image_msg):
        # Update last image time (for reset detection)
        self.last_image_time = rospy.Time.now()
        
        # Check for simulation reset periodically
        self.check_for_reset()
        
        if not self.recording:
            return
        
        # Rate limiting - only collect at 10 fps
        current_time = rospy.Time.now()
        time_since_last_sample = (current_time - self.last_sample_time).to_sec()
        if time_since_last_sample < self.min_time_between_samples:
            return  # Skip this frame
        
        self.last_sample_time = current_time
        
        # Skip if we don't have scan data yet
        if self.latest_scan is None:
            rospy.logwarn_throttle(1.0, "Waiting for scan data before recording...")
            return
        
        # Verify scan has data
        if len(self.latest_scan.ranges) == 0:
            rospy.logwarn_throttle(1.0, "Scan has no ranges, skipping...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion error: {e}")
            return

        # Generate filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"img_{timestamp}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)

        # Save image
        success = cv2.imwrite(image_path, cv_image)
        if not success:
            rospy.logerr(f"Failed to write image to {image_path}")
            return

        # Save to CSV
        rel_image_path = os.path.join('images', image_filename)
        
        v = self.latest_twist.linear.x
        w = self.latest_twist.angular.z
        
        # Process Scan: Take min range or a subset, or save raw
        # Saving raw ranges as a string to be parsed later
        # We replace 'inf' with a large number
        scan_ranges = list(self.latest_scan.ranges)
        scan_ranges = [30.0 if x == float('inf') else x for x in scan_ranges]
        scan_str = str(scan_ranges)

        try:
            with open(self.csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([rel_image_path, v, w, scan_str])
        except Exception as e:
            rospy.logerr(f"Failed to write to CSV: {e}")
            return

        self.count += 1
        if self.count % 10 == 0:
            rospy.loginfo(f"Collected {self.count} samples (v={v:.2f}, w={w:.2f})")

if __name__ == '__main__':
    try:
        dc = DataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
