#!/usr/bin/env python3
"""
Quick diagnostic script to check if controller system is working.
Run this to see exactly what's happening with topics.
"""

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

joy_received = False
cmd_vel_received = False

def joy_callback(msg):
    global joy_received
    joy_received = True
    print(f"✓ /joy OK - Axes: {len(msg.axes)}, Buttons: {len(msg.buttons)}")
    if len(msg.axes) >= 2:
        print(f"  Left Stick: X={msg.axes[0]:.2f}, Y={msg.axes[1]:.2f}")

def cmd_vel_callback(msg):
    global cmd_vel_received
    cmd_vel_received = True
    print(f"✓ /B1/cmd_vel OK - Linear: {msg.linear.x:.2f}, Angular: {msg.angular.z:.2f}")

if __name__ == '__main__':
    rospy.init_node('teleop_diagnostic')
    
    print("=="*40)
    print("CONTROLLER DIAGNOSTIC TOOL")
    print("=="*40)
    print("\nListening for 5 seconds...")
    print("Move the LEFT STICK on your controller!\n")
    
    rospy.Subscriber('/joy', Joy, joy_callback)
    rospy.Subscriber('/B1/cmd_vel', Twist, cmd_vel_callback)
    
    rospy.sleep(5)
    
    print("\n" + "=="*40)
    print("RESULTS:")
    print("=="*40)
    
    if not joy_received:
        print("✗ NO JOY DATA - Check:")
        print("  1. Controller plugged in?")
        print("  2. Run: ls /dev/input/js*")
        print("  3. Run: rosnode info /joy_node")
        print("  4. Check joy_node is running: rosnode list | grep joy")
    else:
        print("✓ Joy node working")
    
    if not cmd_vel_received:
        print("✗ NO CMD_VEL - Check:")
        print("  1. Is teleop_ps4 node running?")
        print("  2. Run: rosnode info /teleop_ps4")
        print("  3. Check logs: rosnode list | grep teleop")
    else:
        print("✓ Teleop node working")
    
    if joy_received and cmd_vel_received:
        print("\n✓✓✓ EVERYTHING WORKING! ✓✓✓")
        print("If robot still not moving, check:")
        print("  - Is Gazebo simulation running?")
        print("  - Is robot paused?")
        print("  - Run: rostopic info /B1/cmd_vel")
