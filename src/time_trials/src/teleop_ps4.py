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
        
        # PS4 Controller mapping
        # Left stick vertical (axis 1) -> Linear X
        # Left stick horizontal (axis 0) -> Angular Z
        
        twist.linear.x = data.axes[1] * self.linear_scale
        twist.angular.z = data.axes[0] * self.angular_scale

        self.pub_vel.publish(twist)

if __name__ == '__main__':
    try:
        teleop = PS4Teleop()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
