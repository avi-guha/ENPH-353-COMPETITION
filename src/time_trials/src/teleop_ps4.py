#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

class PS4Teleop:
    def __init__(self):
        rospy.init_node('teleop_ps4')

        # Parameters
        self.linear_scale = rospy.get_param('~linear_scale', 0.5)
        self.angular_scale = rospy.get_param('~angular_scale', 1.0)
        self.deadzone = rospy.get_param('~deadzone', 0.1)  # Ignore small movements
        self.publish_rate = rospy.get_param('~publish_rate', 20.0)  # Hz
        self.joy_timeout = rospy.get_param('~joy_timeout', 1.0)  # Seconds without joy messages before stopping
        
        # SMOOTHING PARAMETERS - Tuned for data collection
        self.smoothing_alpha = rospy.get_param('~smoothing_alpha', 0.3)  # 0.0 = no smoothing, 1.0 = instant
        self.max_linear_accel = rospy.get_param('~max_linear_accel', 1.0)  # m/s² 
        self.max_angular_accel = rospy.get_param('~max_angular_accel', 2.0)  # rad/s²

        # State tracking
        self.last_joy_msg = None
        self.last_joy_time = None
        self.last_twist = Twist()
        
        # Smoothed command tracking (for EMA filter)
        self.smoothed_linear = 0.0
        self.smoothed_angular = 0.0
        
        self.joy_connected = False
        self.enabled = True  # Can be disabled to let other controllers take over
        self.published_stop = False  # Track if we've already published a stop command
        self.last_pub_check_time = rospy.Time.now()
        self.pub_check_interval = 2.0  # Check publisher connection every 2 seconds
        
        # Publishers and subscribers (use absolute topics to be robust)
        self.cmd_vel_topic = '/B1/cmd_vel'
        self.pub_vel = self.create_cmd_vel_publisher()
        self.pub_active = rospy.Publisher('/teleop_active', Bool, queue_size=1, latch=True)
        
        # Subscribe with tcp_nodelay to reduce latency
        rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/teleop_enable', Bool, self.enable_callback, queue_size=1)

        # Wait a bit for joy_node to start
        rospy.loginfo("PS4 Teleop Node Started (SMOOTH MODE)")
        rospy.loginfo(f"  Linear scale: {self.linear_scale}")
        rospy.loginfo(f"  Angular scale: {self.angular_scale}")
        rospy.loginfo(f"  Deadzone: {self.deadzone}")
        rospy.loginfo(f"  Publish rate: {self.publish_rate} Hz")
        rospy.loginfo(f"  Smoothing alpha: {self.smoothing_alpha} (lower = smoother)")
        rospy.loginfo(f"  Max linear accel: {self.max_linear_accel} m/s²")
        rospy.loginfo(f"  Max angular accel: {self.max_angular_accel} rad/s²")
        rospy.loginfo(f"  Joy timeout: {self.joy_timeout}s")
        rospy.loginfo("Waiting for joystick input on /joy...")
        rospy.loginfo("Publish to /teleop_enable (Bool) to enable/disable this controller")
    
    def create_cmd_vel_publisher(self):
        """Create or recreate the cmd_vel publisher"""
        return rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1, latch=False)
    
    def check_publisher_connection(self):
        """Check if publisher is connected and recreate if needed"""
        current_time = rospy.Time.now()
        if (current_time - self.last_pub_check_time).to_sec() < self.pub_check_interval:
            return
        
        self.last_pub_check_time = current_time
        
        # Check if we have subscribers
        num_connections = self.pub_vel.get_num_connections()
        
        # If we have no connections and we're trying to control, warn
        if num_connections == 0 and self.last_joy_msg is not None:
            rospy.logwarn_throttle(10.0, 
                f"No subscribers to {self.cmd_vel_topic}. Simulation may have reset. Recreating publisher...")
            # Unregister old publisher
            self.pub_vel.unregister()
            # Create new publisher
            self.pub_vel = self.create_cmd_vel_publisher()
            # Small delay to let it register
            rospy.sleep(0.1)

    def enable_callback(self, msg):
        """Enable or disable the teleop controller"""
        if msg.data != self.enabled:
            self.enabled = msg.data
            status = "ENABLED" if self.enabled else "DISABLED"
            rospy.loginfo(f"Teleop controller {status}")
            if not self.enabled:
                # Publish stop command when disabled
                self.pub_vel.publish(Twist())
                self.published_stop = True
    
    def joy_callback(self, data):
        """Process joystick input with deadzone and smoothing"""
        # Check if we have enough axes
        if len(data.axes) < 2:
            if not self.joy_connected:
                rospy.logwarn(f"Not enough axes! Got {len(data.axes)}, need at least 2")
            return
        
        # Log connection on first valid message
        if not self.joy_connected:
            rospy.loginfo("✓ Joystick connected and ready")
            self.joy_connected = True
        
        # Store the latest joystick message with timestamp
        self.last_joy_msg = data
        self.last_joy_time = rospy.Time.now()
        self.published_stop = False  # Reset stop flag when we get new input
    
    def apply_deadzone(self, value):
        """Apply deadzone to joystick input to prevent drift"""
        if abs(value) < self.deadzone:
            return 0.0
        # Scale the value so deadzone threshold maps to 0 and 1.0 stays at 1.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)
    
    def apply_smoothing(self, target, current, alpha):
        """Exponential moving average smoothing"""
        # alpha = 1.0 means instant (no smoothing)
        # alpha = 0.0 means never change (full smoothing)
        # alpha = 0.3 means 30% new value, 70% old value (moderate smoothing)
        return alpha * target + (1 - alpha) * current
    
    def rate_limit(self, target, current, max_change, dt):
        """Limit rate of change (acceleration limiting)"""
        max_delta = max_change * dt
        delta = target - current
        if abs(delta) > max_delta:
            return current + max_delta * (1 if delta > 0 else -1)
        return target
    
    def run(self):
        """Main control loop with consistent publishing rate"""
        rate = rospy.Rate(self.publish_rate)
        
        while not rospy.is_shutdown():
            # Periodically check publisher connection (for sim resets)
            self.check_publisher_connection()
            
            twist = Twist()
            joy_active = False
            
            # Check if teleop is enabled
            if not self.enabled:
                # Publish stop once if not already done
                if not self.published_stop:
                    self.pub_vel.publish(twist)
                    self.published_stop = True
                self.pub_active.publish(Bool(data=False))
                rate.sleep()
                continue
            
            # Check for joystick timeout (watchdog)
            if self.last_joy_time is not None:
                time_since_joy = (rospy.Time.now() - self.last_joy_time).to_sec()
                
                if time_since_joy > self.joy_timeout:
                    # No recent joystick input - publish stop and warn
                    if self.joy_connected and not self.published_stop:
                        rospy.logwarn_throttle(5.0, 
                            f"No joystick input for {time_since_joy:.1f}s - sending stop command")
                        self.pub_vel.publish(twist)
                        self.published_stop = True
                        joy_active = False
                else:
                    joy_active = True
            
            if self.last_joy_msg is not None and joy_active:
                # PS4 Controller mapping - LEFT STICK ONLY
                # Left stick vertical (axis 1) -> Linear X (forward/backward)
                # Left stick horizontal (axis 0) -> Angular Z (turn left/right)
                
                raw_linear = self.last_joy_msg.axes[1]
                raw_angular = self.last_joy_msg.axes[0]
                
                # Apply deadzone and scaling to get target velocities
                target_linear = self.apply_deadzone(raw_linear) * self.linear_scale
                target_angular = self.apply_deadzone(raw_angular) * self.angular_scale
                
                # Calculate dt for rate limiting
                dt = 1.0 / self.publish_rate
                
                # Apply exponential smoothing
                self.smoothed_linear = self.apply_smoothing(
                    target_linear, self.smoothed_linear, self.smoothing_alpha)
                self.smoothed_angular = self.apply_smoothing(
                    target_angular, self.smoothed_angular, self.smoothing_alpha)
                
                # Apply rate limiting (acceleration capping)
                twist.linear.x = self.rate_limit(
                    self.smoothed_linear, self.last_twist.linear.x, 
                    self.max_linear_accel, dt)
                twist.angular.z = self.rate_limit(
                    self.smoothed_angular, self.last_twist.angular.z,
                    self.max_angular_accel, dt)
                
                # Log only when values change significantly (reduce spam)
                if (abs(twist.linear.x - self.last_twist.linear.x) > 0.05 or 
                    abs(twist.angular.z - self.last_twist.angular.z) > 0.05):
                    if abs(twist.linear.x) > 0.01 or abs(twist.angular.z) > 0.01:
                        rospy.loginfo_throttle(0.5, 
                            f"Control: v={twist.linear.x:.2f}, w={twist.angular.z:.2f}")
                
                self.last_twist = twist
                self.pub_vel.publish(twist)
                self.published_stop = False
            
            # Publish active status
            self.pub_active.publish(Bool(data=joy_active and self.enabled))
            rate.sleep()

if __name__ == '__main__':
    try:
        teleop = PS4Teleop()
        teleop.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt, shutting down teleop")
