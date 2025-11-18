#!/usr/bin/env python3

"""
Image Processing for Line Following
Tasks:
1. Display the image from the camera in a window. 
2. Binarize the image to isolate the lines. Display the binary image in a separate window.
3. Using the binarized image, draw a green line in the center of the detected path and display this in a separate window.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageProcessor:
    """Process camera images for line following"""
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('image_processor', anonymous=True)
        
        # Create CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Subscribe to camera topic
        # Adjust topic name based on your robot configuration
        self.image_sub = rospy.Subscriber(
            '/B1/rrbot/camera1/image_raw',  # Change if using different robot
            Image,
            self.image_callback
        )
        
        # Binary threshold value (adjust based on lighting)
        self.binary_threshold = 180
        
        print("Image Processor Started!")
        print("Windows will show:")
        print("  1. Original Camera Image")
        print("  2. Binary Image (lines isolated)")
        print("  3. Center Line Detection")
        print("\nPress 'q' to quit")
        
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # TASK 1: Display the original image
            cv2.imshow("1. Original Camera Image", cv_image)
            
            # TASK 2: Binarize the image to isolate the lines
            binary_image = self.binarize_image(cv_image)
            cv2.imshow("2. Binary Image (Lines Isolated)", binary_image)
            
            # TASK 3: Draw green line in center of detected path
            center_image = self.draw_center_line(cv_image.copy(), binary_image)
            cv2.imshow("3. Center Line Detection", center_image)
            
            # Wait for key press (1ms) - required for imshow to work
            cv2.waitKey(1)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def binarize_image(self, image):
        """
        TASK 2: Binarize the image to isolate white/yellow lines
        Only processes the bottom 60% of the image
        
        Args:
            image: BGR image from camera
            
        Returns:
            binary: Binary image where white pixels = lines, black = road
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create black image (all zeros)
        binary = np.zeros_like(gray)
        
        # Only process bottom 60% of the image
        bottom_60_start = int(height * 0.4)
        bottom_region = gray[bottom_60_start:, :]
        
        # Apply threshold to detect bright lines (white/yellow)
        # Lines are bright (high pixel values), road is dark
        _, binary_bottom = cv2.threshold(bottom_region, self.binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Place the binarized bottom 60% into the full binary image
        binary[bottom_60_start:, :] = binary_bottom
        
        return binary
    
    def draw_center_line(self, image, binary_image):
        """
        TASK 3: Draw a green line in the center of the detected path
        Only analyzes the bottom 60% of the image
        
        Strategy:
        1. Divide bottom 60% into horizontal sections
        2. For each section, find the leftmost and rightmost white pixels
        3. Calculate the center point between them
        4. Draw green line connecting the center points
        
        Args:
            image: Original BGR image to draw on
            binary_image: Binary image with detected lines
            
        Returns:
            image: Image with green center line drawn
        """
        height, width = binary_image.shape
        
        # Only analyze bottom 60% of the image
        bottom_60_start = int(height * 0.4)
        
        # Number of horizontal sections to analyze
        num_sections = 20
        section_height = (height - bottom_60_start) // num_sections
        
        # Store center points
        center_points = []
        
        # Analyze each horizontal section in the bottom 60%
        for i in range(num_sections):
            y_start = bottom_60_start + i * section_height
            y_end = bottom_60_start + (i + 1) * section_height if i < num_sections - 1 else height
            y_center = (y_start + y_end) // 2
            
            # Get the row of pixels in the middle of this section
            section_row = binary_image[y_center, :]
            
            # Find white pixels (lines) in this row
            white_pixels = np.where(section_row > 128)[0]
            
            if len(white_pixels) >= 2:
                # Find leftmost and rightmost white pixels
                left_edge = white_pixels[0]
                right_edge = white_pixels[-1]
                
                # Calculate center between the two edges
                center_x = (left_edge + right_edge) // 2
                
                center_points.append((center_x, y_center))
                
                # Draw small markers at the edges (optional - for debugging)
                cv2.circle(image, (left_edge, y_center), 3, (255, 0, 0), -1)  # Blue left
                cv2.circle(image, (right_edge, y_center), 3, (0, 0, 255), -1)  # Red right
                
            elif len(white_pixels) > 0:
                # If only one edge detected, use it as reference
                center_x = int(np.mean(white_pixels))
                center_points.append((center_x, y_center))
        
        # Draw green line connecting all center points
        if len(center_points) > 1:
            for i in range(len(center_points) - 1):
                pt1 = center_points[i]
                pt2 = center_points[i + 1]
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)  # Green line, thickness 3
        
        # Draw circles at center points for better visibility
        for point in center_points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green filled circle
        
        # Draw a horizontal line showing the 60% boundary
        cv2.line(image, (0, bottom_60_start), (width, bottom_60_start), (255, 255, 0), 2)
        
        # Add text overlay
        cv2.putText(image, "Green = Center Line", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "Bottom 60% Only", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return image
    
    def run(self):
        """Main loop - keep processing images until shutdown"""
        try:
            while not rospy.is_shutdown():
                # Process callbacks to update camera feed
                rospy.sleep(0.01)  # Small sleep to allow callbacks to process
                
                # Check if user pressed 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            # Clean up OpenCV windows
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    try:
        processor = ImageProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
