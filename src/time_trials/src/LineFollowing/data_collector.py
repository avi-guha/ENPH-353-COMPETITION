#!/usr/bin/env python3

import rospy
import cv2
import os
import csv
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import datetime

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)

        # Parameters
        self.image_topic = rospy.get_param('~image_topic', '/rrbot/camera1/image_raw')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.data_dir = rospy.get_param('~data_dir', 'data')

        # Create data directory
        self.images_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        self.csv_file_path = os.path.join(self.data_dir, 'log.csv')
        
        # Initialize CSV
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'v', 'w'])

        self.bridge = CvBridge()
        self.count = 0

        # Subscribers
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.cmd_vel_sub = message_filters.Subscriber(self.cmd_vel_topic, Twist)

        # Synchronizer
        # Slop allows for slight timing differences
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.cmd_vel_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        rospy.loginfo(f"Data Collector started. Saving to {self.data_dir}")
        rospy.loginfo(f"Subscribed to {self.image_topic} and {self.cmd_vel_topic}")

    def callback(self, image_msg, twist_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Generate filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"img_{timestamp}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)

        # Save image
        cv2.imwrite(image_path, cv_image)

        # Save to CSV
        # We store the relative path or filename. Storing filename is usually enough if in same dir structure.
        # Let's store relative path from data_dir
        rel_image_path = os.path.join('images', image_filename)
        
        v = twist_msg.linear.x
        w = twist_msg.angular.z

        # Only save if moving? Or save everything? 
        # User said: "90% of driving is usually 'go straight'. If you train on this, your robot will never turn."
        # But for data collection, we just record what the user does.
        # We can filter later or user can drive carefully.
        
        with open(self.csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([rel_image_path, v, w])

        self.count += 1
        if self.count % 50 == 0:
            rospy.loginfo(f"Collected {self.count} samples")

if __name__ == '__main__':
    try:
        dc = DataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
