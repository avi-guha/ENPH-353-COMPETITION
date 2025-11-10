#!/usr/bin/env python3

"""
Utility script to visualize camera binning in real-time
Helps debug and understand how the camera image is processed into state bins
"""

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class BinVisualizer:
    """
    Visualizes the camera image processing and binning
    """
    
    def __init__(self):
        rospy.init_node('bin_visualizer', anonymous=True)
        
        self.bridge = CvBridge()
        self.current_image = None
        self.num_bins_per_row = 10
        
        # Subscribe to camera
        self.image_sub = rospy.Subscriber(
            '/rrbot/camera1/image_raw',
            Image,
            self.image_callback
        )
        
        print("Bin Visualizer started!")
        print("Showing camera feed with bin overlays...")
        print("Press 'q' to quit")
        
    def image_callback(self, msg):
        """Process and display image"""
        try:
            # Convert to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            
            # Process and visualize
            self.visualize(cv_image)
            
        except Exception as e:
            rospy.logerr(f"Error: {e}")
    
    def process_image_to_bins(self, image):
        """
        Same processing as in RL environment
        Returns state vector and processed images for visualization
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold for line detection
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Define regions
        middle_start = height // 3
        middle_end = 2 * height // 3
        bottom_start = middle_end
        bottom_end = height
        
        middle_region = binary[middle_start:middle_end, :]
        bottom_region = binary[bottom_start:bottom_end, :]
        
        # Process bins
        bin_width = width // self.num_bins_per_row
        state = np.zeros(20)
        
        # Middle third
        for i in range(self.num_bins_per_row):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width if i < 9 else width
            bin_region = middle_region[:, bin_start:bin_end]
            white_ratio = np.sum(bin_region > 128) / bin_region.size
            state[i] = 1.0 if white_ratio > 0.05 else 0.0
        
        # Bottom third
        for i in range(self.num_bins_per_row):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width if i < 9 else width
            bin_region = bottom_region[:, bin_start:bin_end]
            white_ratio = np.sum(bin_region > 128) / bin_region.size
            state[10 + i] = 1.0 if white_ratio > 0.05 else 0.0
        
        return state, binary, middle_start, middle_end
    
    def visualize(self, image):
        """Create visualization with multiple views"""
        height, width = image.shape[:2]
        
        # Process image
        state, binary, middle_start, middle_end = self.process_image_to_bins(image)
        
        # Create display image
        display = image.copy()
        
        # Draw region boundaries
        cv2.line(display, (0, middle_start), (width, middle_start), (0, 255, 0), 2)
        cv2.line(display, (0, middle_end), (width, middle_end), (0, 255, 0), 2)
        
        # Draw bin divisions
        bin_width = width // self.num_bins_per_row
        for i in range(1, self.num_bins_per_row):
            x = i * bin_width
            cv2.line(display, (x, middle_start), (x, height), (0, 255, 0), 1)
        
        # Highlight active bins
        for i in range(self.num_bins_per_row):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width if i < 9 else width
            
            # Middle third - blue
            if state[i] > 0.5:
                cv2.rectangle(display, 
                            (bin_start, middle_start), 
                            (bin_end, middle_end), 
                            (255, 0, 0), 3)
                # Add bin number
                cv2.putText(display, f"M{i}", 
                          (bin_start + 5, middle_start + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Bottom third - red
            if state[10 + i] > 0.5:
                cv2.rectangle(display, 
                            (bin_start, middle_end), 
                            (bin_end, height), 
                            (0, 0, 255), 3)
                # Add bin number
                cv2.putText(display, f"B{i}", 
                          (bin_start + 5, middle_end + 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add state vector text overlay
        text_y = 30
        cv2.putText(display, "State Vector:", (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Middle third state
        text_y += 30
        middle_str = "Middle: [" + ", ".join([str(int(s)) for s in state[0:10]]) + "]"
        cv2.putText(display, middle_str, (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        
        # Bottom third state
        text_y += 25
        bottom_str = "Bottom: [" + ", ".join([str(int(s)) for s in state[10:20]]) + "]"
        cv2.putText(display, bottom_str, (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        
        # Add statistics
        text_y += 30
        middle_active = int(np.sum(state[0:10]))
        bottom_active = int(np.sum(state[10:20]))
        stats_str = f"Active bins: Middle={middle_active}, Bottom={bottom_active}"
        cv2.putText(display, stats_str, (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Line position estimate
        text_y += 25
        if bottom_active > 0:
            # Calculate weighted average position
            active_indices = [i for i in range(10) if state[10+i] > 0.5]
            avg_pos = np.mean(active_indices)
            center_offset = avg_pos - 4.5  # 4.5 is center
            position_str = f"Line position: {avg_pos:.1f} (offset: {center_offset:+.1f})"
            
            # Color based on centering
            if abs(center_offset) < 1:
                color = (0, 255, 0)  # Green - well centered
            elif abs(center_offset) < 2:
                color = (0, 255, 255)  # Yellow - slightly off
            else:
                color = (0, 0, 255)  # Red - badly off
                
            cv2.putText(display, position_str, (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(display, "Line position: NOT DETECTED", (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Create binary view with color coding
        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Resize binary view to match display height
        binary_resized = cv2.resize(binary_colored, 
                                   (int(width * 0.3), height))
        
        # Combine views side by side
        combined = np.hstack([display, binary_resized])
        
        # Show
        cv2.imshow('Bin Visualization (Original + Binary)', combined)
        
        # Also show just the region of interest
        roi = image[middle_start:height, :].copy()
        roi_height = height - middle_start
        
        # Draw bin overlays on ROI
        bin_height_middle = (middle_end - middle_start)
        for i in range(self.num_bins_per_row):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width if i < 9 else width
            
            # Middle third in ROI
            if state[i] > 0.5:
                cv2.rectangle(roi, 
                            (bin_start, 0), 
                            (bin_end, bin_height_middle), 
                            (255, 0, 0), 2)
            
            # Bottom third in ROI
            if state[10 + i] > 0.5:
                cv2.rectangle(roi, 
                            (bin_start, bin_height_middle), 
                            (bin_end, roi_height), 
                            (0, 0, 255), 2)
        
        cv2.imshow('Region of Interest (Middle + Bottom Thirds)', roi)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.signal_shutdown("User quit")
    
    def run(self):
        """Main loop"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    visualizer = BinVisualizer()
    visualizer.run()
