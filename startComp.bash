#!/bin/bash

# ENPH 353 Competition Startup Script
# This script starts the simulation, teleop, data collector, and camera viewer

set -e  # Exit on error

echo "========================================="
echo "ENPH 353 Competition - Starting..."
echo "========================================="

# Source ROS workspace
echo "[1/5] Sourcing workspace..."
source devel/setup.bash

# Start Gazebo simulation
echo "[2/5] Starting Gazebo simulation..."
cd src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw &
SIM_PID=$!
cd /home/fizzer/ENPH-353-COMPETITION

echo "      Waiting 8 seconds for simulation to initialize..."
sleep 8

# Check if simulation is still running
if ! kill -0 $SIM_PID 2>/dev/null; then
    echo "ERROR: Simulation failed to start!"
    exit 1
fi

# Launch PS4 Teleop
echo "[3/5] Launching PS4 teleop controller..."
roslaunch time_trials time_trials_launch.launch &
LAUNCH_PID=$!

echo "      Waiting 3 seconds for teleop to initialize..."
sleep 3

# Check if launch succeeded
if ! kill -0 $LAUNCH_PID 2>/dev/null; then
    echo "ERROR: Teleop launch failed!"
    exit 1
fi

# Launch data collector in background
echo "[4/5] Launching data collector..."
rosrun time_trials data_collector.py &
DATA_PID=$!
sleep 1

# Launch camera viewer in foreground
echo "[5/5] Launching camera viewer..."
echo ""
echo "========================================="
echo "Startup Complete!"
echo "========================================="
echo "Camera viewer is running in foreground."
echo "Press 'q' in camera window to quit."
echo "Press Ctrl+C to stop all processes."
echo "========================================="
echo ""

# Run camera viewer in foreground (blocks here)
rosrun time_trials camera_viewer.py

# Cleanup when camera viewer exits
echo ""
echo "Shutting down..."
kill $DATA_PID 2>/dev/null || true
kill $LAUNCH_PID 2>/dev/null || true
kill $SIM_PID 2>/dev/null || true
echo "Done."