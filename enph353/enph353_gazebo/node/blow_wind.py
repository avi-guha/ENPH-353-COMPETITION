#! /usr/bin/env python3

import os
import random
import rospy

rospy.init_node('wind_blower')
rate = rospy.Rate(0.1)

# To send messages directly to Gazebo sim we use OS calls
commands = ['gz topic -p /gazebo/default/wind -m "linear_velocity: {x:10, y:10, z:0}"',
            'gz topic -p /gazebo/default/wind -m "linear_velocity: {x:-10, y:10, z:0}"',
            'gz topic -p /gazebo/default/wind -m "linear_velocity: {x:-10, y:-10, z:0}"',
            'gz topic -p /gazebo/default/wind -m "linear_velocity: {x:10, y:-10, z:0}"']

print("Be careful it's windy out there!")

# Iterate through each command every 5 seconds
while not rospy.is_shutdown():
   # Choose a random direction for the wind to blow
   i = random.choice(range(len(commands)))
   # Blow some wind
   print("Wind: " + str(i) + " -> " + commands[i])
   os.system(commands[i])
   # Wait a while
   rate.sleep()