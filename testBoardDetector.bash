#!/bin/bash

# ENPH 353 Board Detector Test Script
# Launches: Simulation, Score Tracker, Board Detector (NO inference node)
# Use this to manually drive the robot and test clueboard detection

set -e  # Exit on error

echo "========================================="
echo "Board Detector Test - Starting..."
echo "========================================="

cd /home/fizzer/ENPH-353-COMPETITION

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

# Launch score tracker in background (must run from its directory for UI files)
echo "[3/4] Launching score tracker..."
cd /home/fizzer/ENPH-353-COMPETITION/src/enph353/enph353_utils/scripts
python3 ./score_tracker.py &
SCORE_PID=$!
cd /home/fizzer/ENPH-353-COMPETITION
sleep 1

# Launch board detector in background
echo "[4/5] Launching board detector..."
python3 /home/fizzer/ENPH-353-COMPETITION/src/competition/clueboard_detection/board_detector.py &
DETECTOR_PID=$!
sleep 1

# Launch GUI in background
echo "[5/5] Launching GUI..."
python3 /home/fizzer/ENPH-353-COMPETITION/src/competition/gui.py &
GUI_PID=$!

echo ""
echo "========================================="
echo "Board Detector Test Started!"
echo "========================================="
echo "Score tracker, board detector, and GUI running."
echo "Inference node NOT running - drive manually!"
echo ""
echo "To drive the robot manually, open a new terminal and run:"
echo "  rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/B1/cmd_vel"
echo ""
echo "Press Ctrl+C to stop all processes."
echo "========================================="
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $GUI_PID 2>/dev/null || true
    kill $DETECTOR_PID 2>/dev/null || true
    kill $SCORE_PID 2>/dev/null || true
    kill $SIM_PID 2>/dev/null || true
    echo "Done."
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait
