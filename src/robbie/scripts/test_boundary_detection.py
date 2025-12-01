#!/usr/bin/env python3

"""
Test script to visualize road boundary detection
Shows how the robot identifies the two boundary lines and calculates road center
"""

import rospy
import sys
import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from rl_environment import LineFollowingEnv


class BoundaryVisualizer:
    """Visualize boundary detection in real-time"""
    
    def __init__(self):
        rospy.init_node('boundary_visualizer', anonymous=True)
        self.bridge = CvBridge()
        self.current_image = None
        
        # Subscribe to camera
        self.image_sub = rospy.Subscriber(
            '/B1/rrbot/camera1/image_raw',
            Image,
            self.image_callback
        )
        
        # Create environment for processing
        self.env = LineFollowingEnv()
        
        print("Boundary Visualizer Started!")
        print("This will show:")
        print("  1. Original camera view")
        print("  2. Binary threshold")
        print("  3. Detected boundaries (Hough lines)")
        print("  4. Road center calculation")
        print("\nPress 'q' to quit")
        
    def image_callback(self, msg):
        """Process incoming images"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error: {e}")
    
    def visualize_boundary_detection(self, image):
        """Visualize the boundary detection process"""
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Define regions
        middle_start = height // 3
        middle_end = 2 * height // 3
        bottom_start = middle_end
        
        # Extract bottom third for visualization
        bottom_region = binary[bottom_start:, :]
        
        # Create visualization image
        vis_image = image.copy()
        
        # Draw region boundaries
        cv2.line(vis_image, (0, middle_start), (width, middle_start), (0, 255, 0), 2)
        cv2.line(vis_image, (0, middle_end), (width, middle_end), (0, 255, 0), 2)
        cv2.putText(vis_image, "TOP THIRD", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_image, "MIDDLE THIRD (Future)", (10, middle_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_image, "BOTTOM THIRD (Current)", (10, middle_end + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detect edges in bottom region
        edges = cv2.Canny(bottom_region, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                minLineLength=bottom_region.shape[0]//3, maxLineGap=10)
        
        # Draw detected lines
        if lines is not None:
            line_info = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Adjust y coordinates to full image
                y1_full = y1 + bottom_start
                y2_full = y2 + bottom_start
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 0:
                    verticality = abs(y2 - y1) / length
                else:
                    verticality = 0
                
                avg_x = (x1 + x2) / 2
                strength = length * verticality
                
                # Draw all detected lines in gray
                cv2.line(vis_image, (x1, y1_full), (x2, y2_full), (128, 128, 128), 1)
                
                line_info.append({
                    'x': avg_x,
                    'strength': strength,
                    'coords': (x1, y1_full, x2, y2_full)
                })
            
            # Sort by strength and highlight top 2
            if len(line_info) >= 2:
                line_info.sort(key=lambda l: l['strength'], reverse=True)
                
                # Draw strongest two lines in color
                left_line = min(line_info[0], line_info[1], key=lambda l: l['x'])
                right_line = max(line_info[0], line_info[1], key=lambda l: l['x'])
                
                # Draw left boundary (blue)
                x1, y1, x2, y2 = left_line['coords']
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                cv2.putText(vis_image, "LEFT", (int(left_line['x']), middle_end + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw right boundary (red)
                x1, y1, x2, y2 = right_line['coords']
                cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                cv2.putText(vis_image, "RIGHT", (int(right_line['x']), middle_end + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Calculate and draw center
                center_x = (left_line['x'] + right_line['x']) / 2
                cv2.line(vis_image, (int(center_x), middle_end), 
                        (int(center_x), height), (0, 255, 0), 3)
                cv2.putText(vis_image, "CENTER", (int(center_x) - 30, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw bin overlay
                bin_width = width // 10
                for i in range(10):
                    x = i * bin_width
                    cv2.line(vis_image, (x, middle_end), (x, height), (255, 255, 0), 1)
                
                # Highlight the center bin
                center_bin = int(center_x / bin_width)
                center_bin = max(0, min(9, center_bin))
                bin_start = center_bin * bin_width
                bin_end = (center_bin + 1) * bin_width
                cv2.rectangle(vis_image, (bin_start, middle_end), (bin_end, height),
                             (0, 255, 255), 2)
                cv2.putText(vis_image, f"BIN {center_bin}", (bin_start, middle_end - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return vis_image, binary, edges
    
    def run(self):
        """Main visualization loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            if self.current_image is not None:
                # Visualize boundary detection
                vis_image, binary, edges = self.visualize_boundary_detection(self.current_image)
                
                # Display images
                cv2.imshow("1. Original + Boundary Detection", vis_image)
                cv2.imshow("2. Binary Threshold", binary)
                
                # Show edges in bottom third for debugging
                height = self.current_image.shape[0]
                middle_end = 2 * height // 3
                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cv2.putText(edges_color, "Canny Edges (Bottom Third)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("3. Edge Detection", edges_color)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            rate.sleep()
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        visualizer = BoundaryVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nVisualization stopped")
        cv2.destroyAllWindows()
