#!/usr/bin/env python3
"""
Diagnostic script to check available ROS topics and their types.
Run this while the simulation is running to verify topic names.
"""

import rospy
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
    print(f"✓ Received scan with {len(msg.ranges)} samples")
    print(f"  First 10 ranges: {list(msg.ranges[:10])}")
    rospy.signal_shutdown("Test complete")

if __name__ == '__main__':
    rospy.init_node('topic_checker', anonymous=True)
    
    print("Checking for scan topics...")
    print("This script will test various possible scan topic names.\n")
    
    # Try different possible topic names
    possible_topics = [
        '/scan',
        '/B1/scan',
        '/rrbot/laser/scan',
        '/head_hokuyo_sensor/scan',
    ]
    
    print("Attempting to subscribe to scan topics...")
    for topic in possible_topics:
        try:
            print(f"\nTrying: {topic}")
            sub = rospy.Subscriber(topic, LaserScan, scan_callback)
            rospy.sleep(2.0)  # Wait 2 seconds for a message
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n--- Topic List ---")
    import subprocess
    result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True)
    print(result.stdout)
    
    print("\n--- Looking for scan-related topics ---")
    for line in result.stdout.split('\n'):
        if 'scan' in line.lower() or 'laser' in line.lower() or 'lidar' in line.lower():
            print(f"  → {line}")
    
    print("\nDone. If no scan message received, check:")
    print("1. Is Gazebo running?")
    print("2. Is the robot spawned (B1)?")
    print("3. Check the URDF for the correct topic name")
