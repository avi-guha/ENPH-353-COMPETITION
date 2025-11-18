#!/bin/bash

echo "========================================================"
echo "Image Processing for Line Following"
echo "========================================================"
echo ""
echo "This script will display three windows:"
echo "  1. Original Camera Image"
echo "  2. Binary Image (lines isolated)"
echo "  3. Center Line Detection (green line)"
echo ""
echo "Press 'q' in any window to quit"
echo ""
echo "Make sure the simulation is running!"
echo ""

# Source ROS environment
source ~/ENPH-353-COMPETITION/devel/setup.bash

# Run the image processor
cd /home/fizzer/ENPH-353-COMPETITION/src/time_trials/src/LineFollowing
python3 ImageProcessing.py
