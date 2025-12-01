#!/bin/bash

# Quick fix verification script
# Run this to verify robot is responding

echo "======================================================================"
echo "ROBOT MOVEMENT VERIFICATION"
echo "======================================================================"
echo ""
echo "This script will verify the robot can receive commands."
echo "WATCH GAZEBO - the robot should move forward!"
echo ""
echo "Press Ctrl+C to stop the robot"
echo ""
echo "Starting in 3 seconds..."
sleep 3

source ~/ENPH-353-COMPETITION/devel/setup.bash

echo ""
echo "Publishing movement command to /B1/cmd_vel..."
echo "Robot should be moving FORWARD now!"
echo ""

rostopic pub /B1/cmd_vel geometry_msgs/Twist "linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0" -r 10
