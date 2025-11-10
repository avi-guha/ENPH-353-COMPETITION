#!/usr/bin/env python3

"""
Simple test script to verify robot movement
Publishes velocity commands to test if robot responds
"""

import rospy
from geometry_msgs.msg import Twist
import time

def test_robot_movement():
    """Test if robot responds to velocity commands"""
    
    rospy.init_node('robot_movement_test', anonymous=True)
    
    # Robot name (B1 by default in the simulation)
    robot_name = 'B1'
    
    # Publisher for robot commands (with namespace)
    cmd_pub = rospy.Publisher(f'/{robot_name}/cmd_vel', Twist, queue_size=1)
    
    # Wait for publisher to connect
    print("Waiting for publisher to connect...")
    time.sleep(2)
    
    print("\n" + "="*60)
    print("ROBOT MOVEMENT TEST")
    print("="*60)
    print(f"\nTesting robot: {robot_name}")
    print(f"Publishing to: /{robot_name}/cmd_vel")
    print("\nThis script will test if the robot responds to commands")
    print("Watch the robot in Gazebo!\n")
    
    # Test 1: Move forward
    print("Test 1: Moving forward for 2 seconds...")
    cmd = Twist()
    cmd.linear.x = 0.5
    cmd.angular.z = 0.0
    
    for i in range(20):
        cmd_pub.publish(cmd)
        rospy.sleep(0.1)
    
    print("  ✓ Forward command sent")
    
    # Stop
    cmd = Twist()
    cmd_pub.publish(cmd)
    rospy.sleep(1)
    
    # Test 2: Turn left
    print("\nTest 2: Turning left for 2 seconds...")
    cmd = Twist()
    cmd.linear.x = 0.3
    cmd.angular.z = 1.0
    
    for i in range(20):
        cmd_pub.publish(cmd)
        rospy.sleep(0.1)
    
    print("  ✓ Left turn command sent")
    
    # Stop
    cmd = Twist()
    cmd_pub.publish(cmd)
    rospy.sleep(1)
    
    # Test 3: Turn right
    print("\nTest 3: Turning right for 2 seconds...")
    cmd = Twist()
    cmd.linear.x = 0.3
    cmd.angular.z = -1.0
    
    for i in range(20):
        cmd_pub.publish(cmd)
        rospy.sleep(0.1)
    
    print("  ✓ Right turn command sent")
    
    # Stop
    cmd = Twist()
    cmd_pub.publish(cmd)
    rospy.sleep(1)
    
    # Test 4: Backward
    print("\nTest 4: Moving backward for 2 seconds...")
    cmd = Twist()
    cmd.linear.x = -0.3
    cmd.angular.z = 0.0
    
    for i in range(20):
        cmd_pub.publish(cmd)
        rospy.sleep(0.1)
    
    print("  ✓ Backward command sent")
    
    # Final stop
    cmd = Twist()
    cmd_pub.publish(cmd)
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\nDid the robot move?")
    print("  - YES: Robot is working correctly! ✓")
    print("  - NO: Check the following:")
    print("    1. Is Gazebo running?")
    print("    2. Is the robot spawned in the simulation?")
    print("    3. Check 'rostopic list' for /cmd_vel topic")
    print("    4. Try: rostopic echo /cmd_vel (to see if commands are published)")
    print("="*60)


if __name__ == '__main__':
    try:
        test_robot_movement()
    except rospy.ROSInterruptException:
        print("\nTest interrupted!")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. ROS master is running (roscore or simulation)")
        print("  2. Robot is spawned in Gazebo")
        print("  3. You've sourced the workspace: source ~/ENPH-353-COMPETITION/devel/setup.bash")
