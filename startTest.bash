#!/bin/bash

# ENPH 353 Testing Script
# This script starts the simulation, inference node, and camera viewer

set -e  # Exit on error

echo "========================================="
echo "ENPH 353 Testing - Starting..."
echo "========================================="

# Source ROS workspace
echo "[1/4] Sourcing workspace..."
source devel/setup.bash

# Start Gazebo simulation
echo "[2/4] Starting Gazebo simulation..."
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

# Launch inference node in background
echo "[3/4] Launching inference node..."
python3 /home/fizzer/ENPH-353-COMPETITION/src/competition/line_following/inference_node.py &
INFERENCE_PID=$!
sleep 1

# Launch camera viewer in foreground
echo "[4/4] Launching camera viewer..."
echo ""
echo "========================================="
echo "Startup Complete!"
echo "========================================="
echo "Inference node is running autonomously."
echo "Camera viewer is running in foreground."
echo "Press 'q' in camera window to quit."
echo "Press Ctrl+C to stop all processes."
echo "========================================="
echo ""

# Run camera viewer in foreground (blocks here)
python3 /home/fizzer/ENPH-353-COMPETITION/src/competition/line_following/camera_viewer.py

# Cleanup when camera viewer exits
echo ""
echo "Shutting down..."
kill $INFERENCE_PID 2>/dev/null || true
kill $SIM_PID 2>/dev/null || true
echo "Done."
