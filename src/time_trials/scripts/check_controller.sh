#!/bin/bash

# Quick controller health check script

echo "=== Controller Health Check ==="
echo ""

# Check if joy_node is running
if rosnode info /joy_node &>/dev/null; then
    echo "✓ joy_node is running"
else
    echo "✗ joy_node is NOT running"
fi

# Check if teleop_ps4 is running
if rosnode info /teleop_ps4 &>/dev/null; then
    echo "✓ teleop_ps4 is running"
else
    echo "✗ teleop_ps4 is NOT running"
fi

# Check /joy topic publication rate
echo ""
echo "Checking /joy topic (5 second sample)..."
joy_hz=$(timeout 5 rostopic hz /joy 2>&1 | grep "average rate" | awk '{print $3}')
if [ -n "$joy_hz" ]; then
    echo "✓ /joy publishing at ${joy_hz} Hz"
else
    echo "✗ /joy not publishing or very slow"
fi

# Check /teleop_active status
echo ""
echo "Checking teleop active status..."
teleop_active=$(timeout 2 rostopic echo /teleop_active/data -n 1 2>/dev/null)
if [ "$teleop_active" == "True" ]; then
    echo "✓ Teleop is ACTIVE and controlling the robot"
elif [ "$teleop_active" == "False" ]; then
    echo "⚠ Teleop is INACTIVE (another controller may be active)"
else
    echo "✗ Cannot read /teleop_active topic"
fi

# Check /B1/cmd_vel publishers
echo ""
echo "Publishers to /B1/cmd_vel:"
rostopic info /B1/cmd_vel 2>/dev/null | grep "Publishers:" -A 5 | grep " \* " | sed 's/^/  /'

echo ""
echo "=== Check Complete ==="
