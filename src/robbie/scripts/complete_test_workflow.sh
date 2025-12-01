#!/bin/bash

# Complete Testing Workflow After Camera Update

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     ROAD BOUNDARY DETECTION - COMPLETE TEST WORKFLOW       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 SYSTEM STATUS:"
echo "   ✅ Camera FOV: 80° → 110° (increased by 30°)"
echo "   ✅ Workspace rebuilt"
echo "   ✅ Boundary detection: Find TWO white lines and stay between them"
echo "   ✅ Image processing: Binarized view to detect white lines"
echo ""

echo "⚠️  CRITICAL: Restart Gazebo simulation before testing!"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "WORKFLOW:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "1️⃣  RESTART SIMULATION"
echo "   Terminal 1:"
echo "   $ roslaunch enph353_gazebo worlds.launch"
echo ""
read -p "   → Press Enter when simulation is running..."
echo ""

echo "2️⃣  TEST BOUNDARY DETECTION"
echo "   This will show you:"
echo "   • Wider camera view (110° FOV)"
echo "   • Left boundary (blue line)"
echo "   • Right boundary (red line)"
echo "   • Center position (green line)"
echo "   • Binarized image showing white lines"
echo ""
read -p "   → Press Enter to start visualization..."

# Source ROS
source ~/ENPH-353-COMPETITION/devel/setup.bash
cd /home/fizzer/ENPH-353-COMPETITION/src/robbie/scripts

echo ""
echo "🎥 Starting boundary detection visualization..."
echo "   Press 'q' in the window to continue to next step"
echo ""

python3 test_boundary_detection.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "3️⃣  VERIFY DETECTION RESULTS"
echo ""
echo "   Did you see:"
echo "   ✓ Both white boundaries clearly visible?"
echo "   ✓ Center (green line) between the two boundaries?"
echo "   ✓ Wider field of view compared to before?"
echo "   ✓ Binary threshold showing two distinct white lines?"
echo ""
read -p "   → Type 'yes' if all good, 'no' if issues: " verification

if [ "$verification" != "yes" ]; then
    echo ""
    echo "⚠️  Issues detected. Check:"
    echo "   • Is simulation running?"
    echo "   • Is robot spawned (B1)?"
    echo "   • Are there white lines in camera view?"
    echo "   • Try adjusting robot position manually"
    echo ""
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "4️⃣  TEST ENVIRONMENT UPDATES"
echo "   Verifying:"
echo "   • Camera updates properly"
echo "   • Collision detection works"
echo "   • Timeout termination works"
echo ""
read -p "   → Press Enter to run environment tests..."

python3 test_env_updates.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "5️⃣  READY TO TRAIN!"
echo ""
echo "   Your system is now configured to:"
echo "   • Detect TWO white boundary lines (110° FOV)"
echo "   • Calculate center between them"
echo "   • Learn to stay centered on the road"
echo ""
echo "   Training options:"
echo ""
echo "   A) Quick test (10 episodes):"
echo "      $ python3 train_rl.py --episodes 10"
echo ""
echo "   B) Full training (100+ episodes):"
echo "      $ python3 train_rl.py --episodes 100"
echo ""
echo "   C) Interactive interface:"
echo "      $ python3 quickstart.py"
echo ""

read -p "   → Start training now? (y/n): " start_training

if [ "$start_training" = "y" ]; then
    echo ""
    echo "🚀 Starting training with 100 episodes..."
    echo ""
    python3 train_rl.py --episodes 100
else
    echo ""
    echo "📝 Manual training commands:"
    echo "   cd /home/fizzer/ENPH-353-COMPETITION/src/robbie/scripts"
    echo "   source ~/ENPH-353-COMPETITION/devel/setup.bash"
    echo "   python3 train_rl.py --episodes 100"
    echo ""
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    WORKFLOW COMPLETE                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
