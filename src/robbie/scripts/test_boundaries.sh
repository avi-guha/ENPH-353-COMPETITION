#!/bin/bash

# Test the new boundary detection approach

echo "=================================================="
echo "Road Boundary Detection - Visual Test"
echo "=================================================="
echo ""
echo "This will show you how the robot detects the"
echo "TWO boundary lines and calculates the road center."
echo ""
echo "You should see:"
echo "  • Blue line = LEFT boundary"
echo "  • Red line = RIGHT boundary"  
echo "  • Green line = CALCULATED CENTER (where robot should be)"
echo "  • Yellow box = Active bin (robot's target position)"
echo ""
echo "Press 'q' in the visualization window to quit"
echo ""

# Source ROS environment
source ~/ENPH-353-COMPETITION/devel/setup.bash

cd /home/fizzer/ENPH-353-COMPETITION/src/robbie/scripts

# Run the visualizer
python3 test_boundary_detection.py
