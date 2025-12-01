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
        self.image_received = False
        self.last_image_time = None
        self.reset_detection_time = None
        
        # Subscribe to camera topic
        self.camera_topic = rospy.get_param('~camera_topic', '/B1/rrbot/camera1/image_raw')
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback, 
                                          queue_size=1, buff_size=2**24)
        
        rospy.loginfo(f"Camera Viewer started, subscribing to: {self.camera_topic}")
        rospy.loginfo("Press 'q' in the camera window to quit")
        
    def image_callback(self, msg):
        """Store the latest image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_image_time = rospy.Time.now()
            
            if not self.image_received:
                rospy.loginfo("✓ Camera images received, displaying window...")
                self.image_received = True
                self.reset_detection_time = None  # Reset the flag when images resume
        except Exception as e:
            rospy.logwarn(f"Image conversion error: {e}")
    
    def run(self):
        """Main loop - display camera feed"""
        rate = rospy.Rate(30)  # 30 Hz
        
        rospy.loginfo("Waiting for camera images. Press 'q' in the camera window to quit.")
        
        # Create a placeholder window immediately
        window_created = False
        image_timeout = 3.0  # Seconds without images before detecting potential reset
        
        while not rospy.is_shutdown():
            # Check for simulation reset (no images for a while after having received them)
            if self.image_received and self.last_image_time is not None:
                time_since_image = (rospy.Time.now() - self.last_image_time).to_sec()
                if time_since_image > image_timeout:
                    if self.reset_detection_time is None:
                        rospy.logwarn(f"No camera images for {time_since_image:.1f}s - simulation may have reset")
                        rospy.loginfo("Attempting to reconnect...")
                        self.reset_detection_time = rospy.Time.now()
                        # Recreate subscriber
                        self.image_sub.unregister()
                        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback,
                                                         queue_size=1, buff_size=2**24)
                        self.image_received = False
            
            # Always call waitKey to process window events, even without an image
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                rospy.loginfo("'q' pressed, closing camera viewer...")
                break
            
            if self.current_image is not None:
                # Create window on first image
                if not window_created:
                    cv2.namedWindow("Robot Camera View", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Robot Camera View", 800, 600)
                    window_created = True
                
                cv2.imshow("Robot Camera View", self.current_image)
            else:
                # Log waiting status periodically
                if not self.image_received:
                    rospy.loginfo_throttle(5.0, "Still waiting for camera images...")
            
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        viewer = CameraViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt, shutting down camera viewer")
        cv2.destroyAllWindows()
