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
        self.data_dir = rospy.get_param('~data_dir', 'data')

        # Create data directory
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.images_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        self.csv_file_path = os.path.join(self.data_dir, 'log.csv')
        
        # Initialize CSV
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'v', 'w', 'scan'])

        self.bridge = CvBridge()
        self.count = 0
        self.recording = False
        self.last_button_state = 0
        
        # Store latest messages
        self.latest_twist = Twist()
        self.latest_scan = None

        # Subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback)
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)

        rospy.loginfo(f"Data Collector ready. Saving to {self.data_dir}")
        rospy.loginfo(f"Subscribed to:")
        rospy.loginfo(f"  Image: {self.image_topic}")
        rospy.loginfo(f"  Cmd_vel: {self.cmd_vel_topic}")
        rospy.loginfo(f"  Scan: {self.scan_topic}")
        rospy.loginfo("Press TRIANGLE (Button 2) to start/stop recording.")
        rospy.loginfo("Recording is currently: PAUSED")

    def joy_callback(self, data):
        # Triangle button is usually index 2 (check your controller mapping)
        # We want to toggle on press (0 -> 1 transition)
        try:
            button_state = data.buttons[2]
            if button_state == 1 and self.last_button_state == 0:
                self.recording = not self.recording
                status = "STARTED" if self.recording else "PAUSED"
                rospy.loginfo(f"Recording {status}")
            self.last_button_state = button_state
        except IndexError:
            pass

    def cmd_vel_callback(self, msg):
        self.latest_twist = msg

    def scan_callback(self, msg):
        self.latest_scan = msg

    def image_callback(self, image_msg):
        if not self.recording:
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
        if self.latest_scan:
            scan_ranges = list(self.latest_scan.ranges)
            scan_ranges = [30.0 if x == float('inf') else x for x in scan_ranges]
            scan_str = str(scan_ranges)
        else:
            scan_str = "[]"

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
