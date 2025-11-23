#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class PS4Teleop:
    def __init__(self):
        rospy.init_node('teleop_ps4')

        # Parameters
        self.linear_scale = rospy.get_param('~linear_scale', 0.5)
        self.angular_scale = rospy.get_param('~angular_scale', 1.0)

        # Publishers
        self.pub_vel = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)

        # Subscribers
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        rospy.loginfo("PS4 Teleop Node Started")

    def joy_callback(self, data):
        twist = Twist()
        
        # PS4 Controller mapping - LEFT STICK ONLY
        # Left stick vertical (axis 1) -> Linear X
        # Left stick horizontal (axis 0) -> Angular Z
        
        # Check if we have enough axes
        if len(data.axes) < 2:
            rospy.logwarn(f"Not enough axes! Got {len(data.axes)}, need at least 2")
            return
        
        # Forward/backward on left stick
        twist.linear.x = data.axes[1] * self.linear_scale
        # Left/right turn on left stick
        twist.angular.z = data.axes[0] * self.angular_scale
        ller
        # Only publish if there's actual movement to reduce spam
        if abs(twist.linear.x) > 0.01 or abs(twist.angular.z) > 0.01:
            rospy.loginfo(f"Publishing: v={twist.linear.x:.2f}, w={twist.angular.z:.2f}")
            self.pub_vel.publish(twist)
        else:
            # Still publish to stop the robot
            self.pub_vel.publish(twist)

if __name__ == '__main__':
    try:
        teleop = PS4Teleop()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
