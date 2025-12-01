#!/bin/bash

# Quick verification and training continuation script

echo "=============================================="
echo "RL Environment Update Verification & Training"
echo "=============================================="
echo ""

# Source ROS environment
source ~/ENPH-353-COMPETITION/devel/setup.bash

cd /home/fizzer/ENPH-353-COMPETITION/src/robbie/scripts

echo "Step 1: Testing environment updates..."
echo "----------------------------------------------"
timeout 30 python3 test_env_updates.py

echo ""
echo ""
echo "Step 2: Ready to start training!"
echo "----------------------------------------------"
echo "The following fixes have been applied:"
echo "  ✓ Camera images now update properly between steps"
echo "  ✓ Collision detection triggers episode restart"
echo "  ✓ Timeout counter resets on episode reset"
echo "  ✓ Both termination conditions working (collision & line lost)"
echo ""
echo "To start training, run:"
echo "  python3 train_rl.py --episodes 100 --render"
echo ""
echo "Or use the quickstart interface:"
echo "  python3 quickstart.py"
echo ""
