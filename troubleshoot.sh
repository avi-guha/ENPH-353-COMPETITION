#!/bin/bash

# Troubleshooting script for ENPH 353 Competition

echo "========================================="
echo "ENPH 353 Troubleshooting"
echo "========================================="
echo ""

# Check if roscore is running
echo "[1] Checking ROS master..."
if rostopic list &>/dev/null; then
    echo "    ✓ ROS master is running"
else
    echo "    ✗ ROS master is NOT running"
    echo "    → Start with: roscore &"
    exit 1
fi

# Check for camera topic
echo ""
echo "[2] Checking camera topic..."
if rostopic list | grep -q "/B1/rrbot/camera1/image_raw"; then
    echo "    ✓ Camera topic exists"
    
    # Check publication rate
    echo "    Checking publication rate (5 second sample)..."
    hz_output=$(timeout 5 rostopic hz /B1/rrbot/camera1/image_raw 2>&1 | grep "average rate" | awk '{print $3}')
    if [ -n "$hz_output" ]; then
        echo "    ✓ Camera publishing at ${hz_output} Hz"
    else
        echo "    ✗ Camera topic exists but not publishing"
    fi
else
    echo "    ✗ Camera topic NOT found"
    echo "    → Is the simulation running?"
fi

# Check for joy topic
echo ""
echo "[3] Checking joystick..."
if rostopic list | grep -q "/joy"; then
    echo "    ✓ /joy topic exists"
    
    # Try to read a message
    timeout 2 rostopic echo /joy -n 1 &>/dev/null
    if [ $? -eq 0 ]; then
        echo "    ✓ Joystick is publishing"
    else
        echo "    ⚠ /joy topic exists but not receiving data"
        echo "    → Is your PS4 controller connected?"
    fi
else
    echo "    ✗ /joy topic NOT found"
    echo "    → Is joy_node running?"
fi

# Check for cmd_vel topic
echo ""
echo "[4] Checking cmd_vel topic..."
if rostopic list | grep -q "/B1/cmd_vel"; then
    echo "    ✓ /B1/cmd_vel topic exists"
    
    # Check subscribers
    subs=$(rostopic info /B1/cmd_vel 2>/dev/null | grep "Subscribers:" -A 3 | grep " \* " | wc -l)
    echo "    → ${subs} subscriber(s) to cmd_vel"
    
    # Check publishers
    pubs=$(rostopic info /B1/cmd_vel 2>/dev/null | grep "Publishers:" -A 10 | grep " \* ")
    echo "    Publishers:"
    if [ -n "$pubs" ]; then
        echo "$pubs" | sed 's/^/      /'
    else
        echo "      (none)"
    fi
else
    echo "    ✗ /B1/cmd_vel topic NOT found"
    echo "    → Is the robot spawned in simulation?"
fi

# Check running nodes
echo ""
echo "[5] Checking ROS nodes..."
nodes=$(rosnode list 2>/dev/null)

for node in joy_node teleop_ps4 data_collector camera_viewer; do
    if echo "$nodes" | grep -q "$node"; then
        echo "    ✓ $node is running"
    else
        echo "    ✗ $node is NOT running"
    fi
done

# Check if Gazebo is running
echo ""
echo "[6] Checking Gazebo..."
if pgrep -x "gzserver" > /dev/null; then
    echo "    ✓ Gazebo server is running"
else
    echo "    ✗ Gazebo server is NOT running"
fi

if pgrep -x "gzclient" > /dev/null; then
    echo "    ✓ Gazebo client is running"
else
    echo "    ✗ Gazebo client is NOT running"
fi

echo ""
echo "========================================="
echo "Troubleshooting complete"
echo "========================================="
echo ""
echo "Common fixes:"
echo "  - Camera not working: Wait 10 seconds after sim start"
echo "  - Controller not working: Check PS4 pairing (hold SHARE + PS)"
echo "  - After Ctrl+R: Nodes should auto-reconnect in 2-3 seconds"
echo "  - Still broken: Restart with ./startComp.bash"
echo ""
