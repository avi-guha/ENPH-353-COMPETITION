#!/usr/bin/env python3

"""
Simple camera viewer - displays the robot's camera feed in a window.
"""

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraViewer:
    def __init__(self):
        rospy.init_node('camera_viewer', anonymous=True)
        
        self.bridge = CvBridge()
        self.current_image = None
        
        # Subscribe to camera topic
        camera_topic = rospy.get_param('~camera_topic', '/B1/rrbot/camera1/image_raw')
        rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1)
        
        rospy.loginfo(f"Camera Viewer started, subscribing to: {camera_topic}")
        rospy.loginfo("Press 'q' in the camera window to quit")
        
    def image_callback(self, msg):
        """Store the latest image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Image conversion error: {e}")
    
    def run(self):
        """Main loop - display camera feed"""
        # Create window immediately so it's ready for keypresses
        cv2.namedWindow("Robot Camera View", cv2.WINDOW_NORMAL)
        
        rate = rospy.Rate(30)  # 30 Hz
        
        rospy.loginfo("Camera window ready. Press 'q' to quit.")
        
        while not rospy.is_shutdown():
            if self.current_image is not None:
                cv2.imshow("Robot Camera View", self.current_image)
            
            # Check for keypress (window must be in focus)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                rospy.loginfo("'q' pressed, closing camera viewer...")
                break
                
            rate.sleep()
        
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Camera viewer closed by user")

if __name__ == '__main__':
    try:
        viewer = CameraViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass
