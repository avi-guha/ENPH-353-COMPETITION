#!/bin/bash

# Test script to diagnose controller issues

echo "=== ROS Controller Diagnostics ==="
echo ""

echo "1. Checking if ROS master is running..."
if rostopic list &>/dev/null; then
    echo "   ✓ ROS master is running"
else
    echo "   ✗ ROS master is NOT running"
    echo "   Start the simulation first with ./startComp.bash"
    exit 1
fi

echo ""
echo "2. Checking for joy topic..."
if rostopic list | grep -q "/joy"; then
    echo "   ✓ /joy topic exists"
    echo "   Topic info:"
    rostopic info /joy | sed 's/^/     /'
else
    echo "   ✗ /joy topic NOT found"
    echo "   Available topics:"
    rostopic list | grep -i joy | sed 's/^/     /'
fi

echo ""
echo "3. Checking for cmd_vel topic..."
if rostopic list | grep -q "/B1/cmd_vel"; then
    echo "   ✓ /B1/cmd_vel topic exists"
    echo "   Topic info:"
    rostopic info /B1/cmd_vel | sed 's/^/     /'
else
    echo "   ✗ /B1/cmd_vel topic NOT found"
    echo "   Available cmd_vel topics:"
    rostopic list | grep cmd_vel | sed 's/^/     /'
fi

echo ""
echo "4. Checking for teleop node..."
if rosnode list | grep -q "teleop_ps4"; then
    echo "   ✓ teleop_ps4 node is running"
    echo "   Node info:"
    rosnode info /teleop_ps4 2>/dev/null | head -20 | sed 's/^/     /'
else
    echo "   ✗ teleop_ps4 node NOT running"
    echo "   Running nodes:"
    rosnode list | sed 's/^/     /'
fi

echo ""
echo "5. Testing joystick input (press Ctrl+C to stop)..."
echo "   Move your PS4 controller sticks now..."
timeout 5 rostopic echo /joy --limit 2 2>/dev/null || echo "   ✗ No joystick messages received in 5 seconds"

echo ""
echo "=== Diagnostics complete ==="
