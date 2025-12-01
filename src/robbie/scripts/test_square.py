#!/usr/bin/env python3

"""
Visual feedback test - Makes robot move in a square pattern
Watch Gazebo to verify robot is responding
"""

import rospy
from geometry_msgs.msg import Twist
import sys

def move_square():
    rospy.init_node('square_test', anonymous=True)
    
    robot_name = 'B1'
    cmd_pub = rospy.Publisher(f'/{robot_name}/cmd_vel', Twist, queue_size=10)
    
    print("\n" + "="*60)
    print("VISUAL MOVEMENT TEST - SQUARE PATTERN")
    print("="*60)
    print(f"\nRobot: {robot_name}")
    print(f"Topic: /{robot_name}/cmd_vel")
    print("\nThe robot will move in a square pattern.")
    print("WATCH GAZEBO to see if the robot moves!\n")
    print("Press Ctrl+C to stop\n")
    
    rospy.sleep(2)  # Wait for connection
    
    rate = rospy.Rate(10)  # 10 Hz
    
    try:
        for side in range(4):
            print(f"Side {side + 1}/4: Moving forward...")
            
            # Move forward
            cmd = Twist()
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
            
            for _ in range(30):  # 3 seconds at 10 Hz
                cmd_pub.publish(cmd)
                rate.sleep()
            
            print(f"Side {side + 1}/4: Turning...")
            
            # Turn 90 degrees
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 1.57  # 90 degrees per second
            
            for _ in range(10):  # 1 second
                cmd_pub.publish(cmd)
                rate.sleep()
            
            # Brief pause
            cmd = Twist()
            for _ in range(5):
                cmd_pub.publish(cmd)
                rate.sleep()
        
        # Final stop
        print("\nSquare complete! Stopping robot...")
        cmd = Twist()
        for _ in range(10):
            cmd_pub.publish(cmd)
            rate.sleep()
        
        print("\n" + "="*60)
        print("✓ TEST COMPLETE!")
        print("="*60)
        print("\nDid you see the robot move in Gazebo?")
        print("  YES → Robot is working! You can now train RL agent.")
        print("  NO  → Check TROUBLESHOOTING.md for more help.")
        print("="*60)
        
    except rospy.ROSInterruptException:
        print("\nTest interrupted!")
    except KeyboardInterrupt:
        print("\n\nStopping robot...")
        cmd = Twist()
        cmd_pub.publish(cmd)

if __name__ == '__main__':
    move_square()
