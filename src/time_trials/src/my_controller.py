#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

rospy.init_node('topic_publisher')

# Publisher to move robot
pub_vel = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
# Publisher to start / stop comp
pub_tim = rospy.Publisher('/score_tracker', String, queue_size=1)

rate = rospy.Rate(2)

start_mov = Twist()
start_mov.linear.x = 0.5
start_mov.angular.z = 0.0

stop_mov = Twist()
stop_mov.linear.x = 0.0
stop_mov.angular.z = 0.0

rospy.sleep(1)

# CHANGE THESE LOOK AT COMP NOTES FOR WHAT SPECS YOU NEED TO WRITE / PUBLISH
start_timer = String('team, pass, 0, whatever')
stop_timer = String('team, pass, -1, whatever')

rospy.sleep(1)

pub_tim.publish(start_timer)
pub_vel.publish(start_mov)
rospy.sleep(10)
pub_vel.publish(stop_mov)
pub_tim.publish(stop_timer)

rospy.spin()

#while not rospy.is_shutdown():
#   pub_vel.publish(move)
#   rate.sleep()
