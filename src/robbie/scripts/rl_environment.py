#!/usr/bin/env python3

"""
Reinforcement Learning Environment for Line Following Robot
Processes camera images into bins for state representation
Handles collision detection and reward calculation
"""

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
import time


class LineFollowingEnv:
    """
    Environment wrapper for Gazebo line following task
    State: Camera image divided into 20 bins (10 middle third, 10 bottom third)
    Actions: Discrete steering and speed combinations
    """
    
    def __init__(self, robot_name='B1'):
        rospy.init_node('rl_line_following_env', anonymous=True)
        
        self.robot_name = robot_name
        self.bridge = CvBridge()
        
        # State representation: 20 bins (10 middle + 10 bottom thirds)
        self.num_bins_per_row = 10
        self.state_size = 20  # 10 bins in middle third + 10 bins in bottom third
        
        # Action space: combinations of angular velocity and linear velocity
        # 5 steering options: hard left, left, straight, right, hard right
        # 3 speed options: slow, medium, fast
        self.action_space = self._define_action_space()
        self.num_actions = len(self.action_space)
        
        # Current state
        self.current_image = None
        self.current_state = np.zeros(self.state_size)
        self.collision_detected = False
        self.last_collision_time = 0
        self.image_received = False  # Track if we've received at least one image
        
        # Reward tracking
        self.episode_reward = 0
        self.steps = 0
        self.max_steps = 1000
        
        # ROS subscribers and publishers (with robot namespace)
        self.image_sub = rospy.Subscriber(
            f'/{robot_name}/rrbot/camera1/image_raw', 
            Image, 
            self.image_callback
        )
        self.collision_sub = rospy.Subscriber(
            f'/{robot_name}_Hit',  # Collision topic for this robot
            Bool, 
            self.collision_callback
        )
        self.cmd_pub = rospy.Publisher(
            f'/{robot_name}/cmd_vel',  # Publish to namespaced topic
            Twist, 
            queue_size=1
        )
        
        # Gazebo services for reset
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Starting position - use original spawn position from simulation
        # Robot spawns at x=5.5, y=2.5 in the simulation
        self.start_positions = [
            {'x': 5.5, 'y': 2.5, 'z': 0.2, 'yaw': -1.57},  # Original position
        ]
        
        # Timeout counter for line detection
        self.timeout = 0
        self.max_timeout = 30  # Max frames without line before termination
        
        time.sleep(1)  # Wait for subscribers to initialize
        
    def _define_action_space(self):
        """
        Define discrete action space
        Returns list of (linear_vel, angular_vel) tuples
        """
        actions = []
        # Linear velocities: slow, medium, fast
        linear_vels = [0.2, 0.4, 0.6]
        # Angular velocities: hard left, left, straight, right, hard right
        angular_vels = [1.5, 0.8, 0.0, -0.8, -1.5]
        
        for lin_vel in linear_vels:
            for ang_vel in angular_vels:
                actions.append((lin_vel, ang_vel))
        
        return actions
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            
            # Process image into binned state
            self.current_state = self._process_image_to_bins(cv_image)
            
            # Mark that we've received an image
            self.image_received = True
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def collision_callback(self, msg):
        """Track collision events"""
        if msg.data and (time.time() - self.last_collision_time) > 0.5:
            self.collision_detected = True
            self.last_collision_time = time.time()
    
    def _process_image_to_bins(self, image):
        """
        Process camera image into 20 bins with visualization
        - Middle third (rows 267-533 of 800): 10 bins (future/lookahead)
        - Bottom third (rows 534-800): 10 bins (current position)
        
        NEW APPROACH: Identify TWO LONGEST boundary lines (left and right edges of road)
        Robot should stay BETWEEN these two lines (centered on road)
        
        Returns: binary array of shape (20,) indicating road center position in each bin
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to detect white/yellow lines
        # Lines are typically bright (white/yellow on dark road)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # Define middle and bottom thirds
        middle_start = height // 3
        middle_end = 2 * height // 3
        bottom_start = middle_end
        bottom_end = height
        
        # Extract regions
        middle_region = binary[middle_start:middle_end, :]
        bottom_region = binary[bottom_start:bottom_end, :]
        
        # Divide each region into 10 bins horizontally
        bin_width = width // self.num_bins_per_row
        
        state = np.zeros(20)
        
        # Process MIDDLE THIRD - find two longest lines and center between them
        middle_center_bin = self._find_road_center(middle_region, bin_width)
        if middle_center_bin >= 0:
            state[middle_center_bin] = 1.0
        
        # Process BOTTOM THIRD - find two longest lines and center between them
        bottom_center_bin = self._find_road_center(bottom_region, bin_width)
        if bottom_center_bin >= 0:
            state[10 + bottom_center_bin] = 1.0
        
        # Update timeout counter
        if bottom_center_bin == -1:
            # No road boundaries detected
            self.timeout += 1
        else:
            # Road boundaries detected, reset timeout
            self.timeout = 0
        
        # Display binary threshold view in separate window
        cv2.imshow("Binary Threshold", binary)
        cv2.waitKey(1)
        
        return state
    
    def _find_road_center(self, region, bin_width):
        """
        Find the center position between the two longest/strongest line boundaries
        
        Args:
            region: Binary image region (middle or bottom third)
            bin_width: Width of each bin in pixels
            
        Returns:
            Center bin index (0-9), or -1 if no boundaries found
        """
        height, width = region.shape
        
        # Detect vertical edges using Canny edge detection
        edges = cv2.Canny(region, 50, 150)
        
        # Use Hough Line Transform to find vertical-ish lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                minLineLength=height//3, maxLineGap=10)
        
        if lines is None or len(lines) < 2:
            # Fallback: use simple column-wise white pixel detection
            return self._find_road_center_fallback(region, bin_width)
        
        # Calculate line strength (length and vertical-ness)
        line_info = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # Calculate verticality (prefer vertical lines)
            if length > 0:
                verticality = abs(y2 - y1) / length  # 1.0 = perfectly vertical
            else:
                verticality = 0
            
            # Average x position
            avg_x = (x1 + x2) / 2
            
            # Strength = length * verticality
            strength = length * verticality
            
            line_info.append({
                'x': avg_x,
                'strength': strength,
                'length': length
            })
        
        # Sort by strength and take top 2
        line_info.sort(key=lambda l: l['strength'], reverse=True)
        
        if len(line_info) < 2:
            return self._find_road_center_fallback(region, bin_width)
        
        # Get the two strongest lines
        left_line = min(line_info[0], line_info[1], key=lambda l: l['x'])
        right_line = max(line_info[0], line_info[1], key=lambda l: l['x'])
        
        # Calculate center between the two lines
        center_x = (left_line['x'] + right_line['x']) / 2
        
        # Convert to bin index
        center_bin = int(center_x / bin_width)
        center_bin = max(0, min(9, center_bin))  # Clamp to valid range
        
        return center_bin
    
    def _find_road_center_fallback(self, region, bin_width):
        """
        Fallback method: Find road center by detecting columns with most white pixels
        Assumes the two columns with most white pixels are the road boundaries
        
        Args:
            region: Binary image region
            bin_width: Width of each bin
            
        Returns:
            Center bin index (0-9), or -1 if no boundaries found
        """
        height, width = region.shape
        
        # Count white pixels in each column
        column_sums = np.sum(region > 128, axis=0)
        
        # Smooth the column sums to reduce noise
        if len(column_sums) > 20:
            kernel_size = 15
            kernel = np.ones(kernel_size) / kernel_size
            column_sums = np.convolve(column_sums, kernel, mode='same')
        
        # Find peaks (local maxima) which represent the lines
        peaks = []
        for i in range(1, len(column_sums) - 1):
            if column_sums[i] > column_sums[i-1] and column_sums[i] > column_sums[i+1]:
                if column_sums[i] > height * 0.1:  # At least 10% of pixels
                    peaks.append((i, column_sums[i]))
        
        if len(peaks) < 2:
            # Not enough boundaries detected
            return -1
        
        # Sort peaks by strength and take top 2
        peaks.sort(key=lambda p: p[1], reverse=True)
        left_peak = min(peaks[0][0], peaks[1][0])
        right_peak = max(peaks[0][0], peaks[1][0])
        
        # Calculate center
        center_x = (left_peak + right_peak) / 2
        
        # Convert to bin index
        center_bin = int(center_x / bin_width)
        center_bin = max(0, min(9, center_bin))
        
        return center_bin
    
    def step(self, action_idx):
        """
        Execute action and return next_state, reward, done, info
        
        Args:
            action_idx: Index into action_space
            
        Returns:
            next_state: np.array of shape (20,)
            reward: float
            done: bool
            info: dict
        """
        # Mark that we need a new image
        self.image_received = False
        
        # Execute action
        linear_vel, angular_vel = self.action_space[action_idx]
        self._publish_velocity(linear_vel, angular_vel)
        
        # Wait for new camera image to arrive
        timeout_counter = 0
        max_wait = 50  # 5 seconds max wait (0.1s * 50)
        while not self.image_received and timeout_counter < max_wait:
            rospy.sleep(0.1)
            timeout_counter += 1
        
        if timeout_counter >= max_wait:
            rospy.logwarn("Timeout waiting for camera image!")
        
        # Get next state (updated by image callback)
        next_state = self.current_state.copy()
        
        # Calculate reward
        reward, done, info = self._calculate_reward(next_state, action_idx)
        
        self.steps += 1
        self.episode_reward += reward
        
        # Check if episode should end
        if self.steps >= self.max_steps:
            done = True
            info['max_steps_reached'] = True
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, state, action_idx):
        """
        Calculate reward based on current state
        
        Two termination conditions:
        1. Collision (from collision sensor) - Episode ends with large penalty
        2. Timeout (road boundaries lost for too long) - Episode ends with large penalty
        
        NEW REWARD STRUCTURE:
        State now represents ROAD CENTER position (between two boundary lines)
        - Current position (bottom third): where robot should be NOW
        - Future position (middle third): where road will be (lookahead)
        
        Reward for staying centered on the road between the two boundary lines
        """
        reward = 0.0
        done = False
        info = {}
        
        # TERMINATION CONDITION 1: COLLISION PENALTY
        if self.collision_detected:
            reward = -500.0  # Massive penalty for collision
            done = True
            info['collision'] = True
            info['termination_reason'] = 'collision'
            rospy.logwarn(f"COLLISION DETECTED! Episode ending with penalty: {reward}")
            # Don't reset flag here - let reset() handle it
            return reward, done, info
        
        # TERMINATION CONDITION 2: TIMEOUT (road boundaries lost for too long)
        if self.timeout > self.max_timeout:
            reward = -500.0  # Massive penalty for losing road boundaries
            done = True
            info['timeout'] = True
            info['termination_reason'] = 'road_lost'
            rospy.logwarn(f"ROAD BOUNDARIES LOST! Episode ending with penalty: {reward}")
            return reward, done, info
        
        # Extract current and future road center positions from state
        bottom_bins = state[10:20]  # Current road center (bottom third)
        middle_bins = state[0:10]   # Future road center (middle third)
        
        # Find current road center position (which bin in bottom third)
        current_center = -1
        for i in range(10):
            if bottom_bins[i] == 1:
                current_center = i
                break
        
        # Find future road center position (which bin in middle third)
        future_center = -1
        for i in range(10):
            if middle_bins[i] == 1:
                future_center = i
                break
        
        # CURRENT POSITION REWARDS (heavily weighted)
        # Robot should be at the current road center
        if current_center == -1:
            # No road boundaries detected - HEAVY penalty
            reward -= 50
        elif current_center in [4, 5]:
            # Robot is CENTERED on road (bins 4 or 5) - VERY HIGH reward
            reward += 100
            # Get action details
            linear_vel, angular_vel = self.action_space[action_idx]
            # MASSIVE bonus if moving forward while perfectly centered
            if linear_vel > 0.3 and abs(angular_vel) < 0.5:
                reward += 50
        elif current_center in [3, 6]:
            # Robot is slightly off center - moderate reward
            reward += 10
        elif current_center in [2, 7]:
            # Robot is more off center - small reward
            reward += 2
        else:
            # Robot is near road edges (bins 0, 1, 8, 9) - HEAVY penalty
            # This means robot is close to the boundary lines
            reward -= 30
        
        # FUTURE POSITION REWARDS (predictive adjustment)
        if future_center >= 0:
            # Reward for future road center also being centered in view
            if future_center in [4, 5]:
                reward += 20  # Bonus for road staying centered ahead
            elif future_center in [3, 6]:
                reward += 5
            # Penalty if future road is at edges (road is curving)
            elif future_center in [0, 1, 8, 9]:
                reward -= 10
            
            # Reward for taking corrective action based on future road position
            linear_vel, angular_vel = self.action_space[action_idx]
            # If road is curving left (future center < 4), reward left turn
            if angular_vel > 0.5 and future_center < 4:
                reward += 15
            # If road is curving right (future center > 5), reward right turn
            elif angular_vel < -0.5 and future_center > 5:
                reward += 15
        
        return reward, done, info
    
    def reset(self, random_start=True):
        """
        Reset environment to starting position
        
        Args:
            random_start: If True, randomly select from start_positions
            
        Returns:
            initial_state: np.array of shape (20,)
        """
        # Choose starting position
        if random_start:
            start_pos = self.start_positions[np.random.randint(len(self.start_positions))]
        else:
            start_pos = self.start_positions[0]
        
        # Create model state message
        model_state = ModelState()
        model_state.model_name = self.robot_name
        model_state.pose.position.x = start_pos['x']
        model_state.pose.position.y = start_pos['y']
        model_state.pose.position.z = start_pos['z']
        
        # Set orientation (yaw)
        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, start_pos['yaw'])
        model_state.pose.orientation.x = q[0]
        model_state.pose.orientation.y = q[1]
        model_state.pose.orientation.z = q[2]
        model_state.pose.orientation.w = q[3]
        
        # Reset velocities
        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0
        
        try:
            self.set_state(model_state)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        
        # Stop robot
        self._publish_velocity(0, 0)
        
        # Reset episode tracking
        self.steps = 0
        self.episode_reward = 0
        self.collision_detected = False
        self.timeout = 0  # Reset timeout counter
        self.image_received = False  # Wait for fresh image after reset
        
        # Wait for state to update and get fresh camera image
        timeout_counter = 0
        max_wait = 50  # 5 seconds max wait
        while not self.image_received and timeout_counter < max_wait:
            rospy.sleep(0.1)
            timeout_counter += 1
        
        return self.current_state.copy()
    
    def _publish_velocity(self, linear, angular):
        """Publish velocity command to robot"""
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
    
    def get_state_size(self):
        """Return size of state vector"""
        return self.state_size
    
    def get_action_size(self):
        """Return number of possible actions"""
        return self.num_actions
    
    def render(self, mode='human'):
        """
        Optional: Display current camera view with bin overlays
        """
        if self.current_image is None:
            return
        
        image = self.current_image.copy()
        height, width = image.shape[:2]
        
        # Draw bin divisions
        middle_start = height // 3
        middle_end = 2 * height // 3
        
        # Draw horizontal lines for thirds
        cv2.line(image, (0, middle_start), (width, middle_start), (0, 255, 0), 2)
        cv2.line(image, (0, middle_end), (width, middle_end), (0, 255, 0), 2)
        
        # Draw vertical lines for bins
        bin_width = width // self.num_bins_per_row
        for i in range(1, self.num_bins_per_row):
            x = i * bin_width
            cv2.line(image, (x, middle_start), (x, height), (0, 255, 0), 1)
        
        # Highlight active bins
        for i in range(self.num_bins_per_row):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width if i < 9 else width
            
            # Middle third
            if self.current_state[i] > 0.5:
                cv2.rectangle(image, 
                            (bin_start, middle_start), 
                            (bin_end, middle_end), 
                            (0, 0, 255), 2)
            
            # Bottom third
            if self.current_state[10 + i] > 0.5:
                cv2.rectangle(image, 
                            (bin_start, middle_end), 
                            (bin_end, height), 
                            (255, 0, 0), 2)
        
        cv2.imshow('Line Following View', image)
        cv2.waitKey(1)
    
    def close(self):
        """Cleanup"""
        self._publish_velocity(0, 0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Test environment
    env = LineFollowingEnv()
    
    try:
        state = env.reset()
        print(f"Initial state: {state}")
        
        for _ in range(100):
            # Random action
            action = np.random.randint(env.get_action_size())
            next_state, reward, done, info = env.step(action)
            
            print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")
            env.render()
            
            if done:
                print("Episode finished!")
                state = env.reset()
            else:
                state = next_state
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()
