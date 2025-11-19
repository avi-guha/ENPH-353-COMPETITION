#!/usr/bin/env python3

"""
Step 1: Display live camera feed
"""

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Global variable to store the latest image
current_image = None

def image_callback(msg):
    """Callback function that receives images from camera"""
    global current_image
    bridge = CvBridge()
    
    try:
        # Convert ROS Image message to OpenCV image
        current_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(f"Error: {e}")

def main():
    global current_image
    
    # Initialize ROS node
    rospy.init_node('camera_viewer', anonymous=True)
    
    # Subscribe to camera topic (from robbie.xacro: /B1/rrbot/camera1/image_raw)
    rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, image_callback)
    
    print("Camera viewer started. Press 'q' to quit.")
    
    # Main loop - just display the image
    while not rospy.is_shutdown():
        if current_image is not None:
            # Show the live camera feed
            cv2.imshow("Live Camera Feed", current_image)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
