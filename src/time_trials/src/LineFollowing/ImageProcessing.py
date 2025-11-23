 #!/usr/bin/env python3

"""
Step 4: Color-based lane line detection (HSV filtering).

Two windows:
    1. Live Camera Feed (original)
    2. Lane Lines Binary (white & yellow lines only, ignoring road surface)

Processing pipeline per frame:
    - Convert to HSV color space for better color discrimination
    - Filter for WHITE lines (high brightness, low saturation)
    - Filter for YELLOW lines (hue ~20-40°, high saturation & brightness)
    - Combine masks to detect all lane markings while ignoring brown/tan road
    - Apply morphological operations to clean noise
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Global variable to store the latest image
current_image = None

# Processing parameters (tunable)
BLUR_KERNEL_SIZE = 5  # Light blur to preserve edges (odd kernel)

# HSV color range for WHITE lane lines (strict for clean detection)
WHITE_LOW_HSV = np.array([0, 0, 200])      # High brightness
WHITE_HIGH_HSV = np.array([180, 40, 255])  # Low saturation for white

# HSV color range for YELLOW lane lines (targeting road paint yellow)
YELLOW_LOW_HSV = np.array([15, 60, 160])   # Yellow hue, moderate sat & value
YELLOW_HIGH_HSV = np.array([35, 255, 255]) # Yellow spectrum

bridge = CvBridge()  # Reuse bridge instead of recreating each callback


def image_callback(msg):
    """Lightweight callback: only convert and store latest image.
    
    Heavy processing moved to main loop to avoid blocking subscriber and
    improve perceived frame rate.
    """
    global current_image
    try:
        current_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logwarn(f"Image callback error: {e}")


def process_full_image(image: np.ndarray) -> np.ndarray:
    """Return full image with lane lines detected via HSV color filtering.
    
    Detects WHITE and YELLOW lines using HSV masks, then filters by contour
    area and position. Inspired by mountain line following approach:
    mask -> find contours -> filter -> thicken.
    """
    # Apply light blur to reduce sensor noise
    blurred = cv2.GaussianBlur(image, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    
    # Convert to HSV color space for better color discrimination
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create mask for WHITE lines
    white_mask = cv2.inRange(hsv, WHITE_LOW_HSV, WHITE_HIGH_HSV)
    
    # Create mask for YELLOW lines
    yellow_mask = cv2.inRange(hsv, YELLOW_LOW_HSV, YELLOW_HIGH_HSV)
    
    # Combine both masks (detect either white OR yellow)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Light morphological operations - don't break up the lines
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,
                                     kernel_small, iterations=1)
    # Close gaps in detected lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE,
                                     kernel_close, iterations=2)
    
    # Find contours on the mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Create blank output image
    output_mask = np.zeros_like(combined_mask)
    
    if len(contours) == 0:
        out_bgr = cv2.cvtColor(output_mask, cv2.COLOR_GRAY2BGR)
        return out_bgr
    
    # Filter contours - keep larger features
    min_contour_area = 100  # Lower threshold to catch lane lines
    height, width = combined_mask.shape
    
    # Get image regions for left and right lanes
    left_third = width // 3
    right_third = 2 * width // 3
    
    left_lane_contours = []
    right_lane_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        
        # Get centroid to determine which side of image
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        
        # Categorize as left or right lane based on position
        if cx < left_third:
            left_lane_contours.append((area, cnt))
        elif cx > right_third:
            right_lane_contours.append((area, cnt))
    
    # Keep only the largest contour from each side (the actual lane lines)
    if left_lane_contours:
        largest_left = max(left_lane_contours, key=lambda x: x[0])[1]
        cv2.drawContours(output_mask, [largest_left], -1, 255, thickness=7)
    
    if right_lane_contours:
        largest_right = max(right_lane_contours, key=lambda x: x[0])[1]
        cv2.drawContours(output_mask, [largest_right], -1, 255, thickness=7)
    
    # Additional thickening for better visibility
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    output_mask = cv2.dilate(output_mask, kernel_dilate, iterations=2)
    
    # Convert to BGR for uniform display
    out_bgr = cv2.cvtColor(output_mask, cv2.COLOR_GRAY2BGR)
    return out_bgr


def main():
    global current_image
    
    # Initialize ROS node
    rospy.init_node('camera_viewer', anonymous=True)
    
    # Subscribe to camera topic (from robbie.xacro: /B1/rrbot/camera1/image_raw)
    rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, image_callback,
                     queue_size=1)
    
    print("Camera processing started. Windows: Original & Full Binary. "
          "Press 'q' to quit.")
    
    # Main loop: process latest frame (drops older frames if backlog)
    # for better responsiveness
    while not rospy.is_shutdown():
        if current_image is not None:
            # Copy to avoid race while new image arrives
            frame = current_image.copy()
            
            # Show original feed
            cv2.imshow("Original Camera Feed", frame)
            
            # Process and show binary
            processed = process_full_image(frame)
            cv2.imshow("Full Image Binary", processed)
        
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
 