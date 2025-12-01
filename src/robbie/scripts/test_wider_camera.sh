#!/bin/bash

echo "========================================================"
echo "Camera FOV Update - Testing Guide"
echo "========================================================"
echo ""
echo "✅ Camera FOV increased: 80° → 110° (+30°)"
echo "✅ Workspace rebuilt successfully"
echo ""
echo "⚠️  IMPORTANT: You must restart the Gazebo simulation"
echo "   for the camera changes to take effect!"
echo ""
echo "========================================================"
echo "Steps to Test:"
echo "========================================================"
echo ""
echo "1. RESTART SIMULATION"
echo "   - Stop current simulation (Ctrl+C in simulation terminal)"
echo "   - Run: roslaunch enph353_gazebo worlds.launch"
echo ""
echo "2. TEST BOUNDARY DETECTION (run this script)"
echo "   - This will visualize the boundary detection"
echo "   - You should see WIDER camera view"
echo "   - Both white lines should be visible"
echo ""
echo "3. START TRAINING"
echo "   - python3 train_rl.py --episodes 100"
echo ""
echo "========================================================"
echo ""
read -p "Press Enter to test boundary detection (simulation must be running)..."

echo ""
echo "Starting boundary detection visualization..."
echo "Look for:"
echo "  • WIDER camera view (110° instead of 80°)"
echo "  • Blue line = Left boundary"
echo "  • Red line = Right boundary"
echo "  • Green line = Center (robot should stay here)"
echo ""
echo "Press 'q' in visualization window to quit"
echo ""

# Source ROS environment
source ~/ENPH-353-COMPETITION/devel/setup.bash

cd /home/fizzer/ENPH-353-COMPETITION/src/robbie/scripts

# Run boundary visualization
python3 test_boundary_detection.py
