# 🔧 Robot Not Moving - Troubleshooting Guide

## Quick Fix Checklist

### 1. ✅ Rebuild the workspace (IMPORTANT - URDF was updated!)
```bash
cd ~/ENPH-353-COMPETITION
catkin_make
source devel/setup.bash
```

### 2. ✅ Start the simulation properly
```bash
# Terminal 1: Start simulation
cd ~/ENPH-353-COMPETITION/src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw
```

**Wait** for Gazebo to fully load (you should see the world and robot).

### 3. ✅ Test robot movement
```bash
# Terminal 2: Test if robot can move
cd ~/ENPH-353-COMPETITION/src/robbie/scripts
source ~/ENPH-353-COMPETITION/devel/setup.bash
./test_movement.py
```

Watch the robot in Gazebo - it should move forward, turn left, turn right, and backward.

### 4. ✅ If robot still doesn't move, check topics
```bash
# Check if /cmd_vel topic exists
rostopic list | grep cmd_vel

# Try publishing manually
rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.5
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0" -r 10
```

The robot should move forward if this works.

---

## Common Issues and Solutions

### Issue 1: "Robot spawns but doesn't respond to commands"

**Cause**: URDF missing `base_link` (FIXED in latest version)

**Solution**: 
```bash
cd ~/ENPH-353-COMPETITION
catkin_make clean  # Clean old build
catkin_make        # Rebuild with fixed URDF
source devel/setup.bash
```

Then restart the simulation.

---

### Issue 2: "No /cmd_vel topic found"

**Cause**: Robot not properly spawned or controller plugin not loaded

**Solution**:
```bash
# Check if robot is in simulation
rosservice call /gazebo/get_world_properties

# Check robot name
rosservice call /gazebo/get_model_state "model_name: 'R1'"

# If robot not spawned, check launch files
roslaunch robbie rl_training.launch
```

---

### Issue 3: "Simulation won't start"

**Cause**: ROS/Gazebo processes still running from previous session

**Solution**:
```bash
# Kill all ROS/Gazebo processes
pkill -f gzserver
pkill -f gzclient
killall -9 rosout roslaunch rosmaster gzserver gzclient

# Wait a few seconds, then restart
cd ~/ENPH-353-COMPETITION/src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw
```

---

### Issue 4: "RL environment connects but robot doesn't move"

**Cause**: Topic name mismatch or rate too slow

**Check**:
1. Run `rostopic hz /cmd_vel` - should show ~10 Hz when RL is running
2. Run `rostopic echo /cmd_vel` - should show velocity commands

**Solution** (if needed):
The environment publishes at 10 Hz (every 0.1 seconds). If commands aren't reaching robot:

```bash
# Check for topic remapping issues
rostopic info /cmd_vel

# Should show:
# Publishers:
#  * /rl_line_following_env (or similar)
# Subscribers:
#  * /gazebo (or robot controller)
```

---

### Issue 5: "Camera not working"

**Check camera topic**:
```bash
# List camera topics
rostopic list | grep camera

# Should see: /rrbot/camera1/image_raw

# Check if images are being published
rostopic hz /rrbot/camera1/image_raw

# View camera feed
rqt_image_view
# Then select /rrbot/camera1/image_raw from dropdown
```

---

## Step-by-Step Complete Restart

If nothing works, do a complete restart:

```bash
# 1. Kill everything
pkill -f gzserver; pkill -f gzclient; killall -9 rosout roslaunch rosmaster

# 2. Clean and rebuild
cd ~/ENPH-353-COMPETITION
catkin_make clean
catkin_make

# 3. Source workspace
source devel/setup.bash

# 4. Start simulation (wait for it to fully load!)
cd src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw

# Wait ~30 seconds for Gazebo to fully initialize

# 5. In NEW terminal, test robot
cd ~/ENPH-353-COMPETITION/src/robbie/scripts
source ~/ENPH-353-COMPETITION/devel/setup.bash
./test_movement.py

# 6. If that works, try RL training
./quickstart.py train --episodes 10 --render
```

---

## Debugging Commands

```bash
# 1. Check ROS is running
rostopic list

# 2. Check robot exists in Gazebo
rosservice call /gazebo/get_world_properties
rosservice call /gazebo/get_model_state "model_name: 'R1'"

# 3. Check topics are connected
rostopic info /cmd_vel
rostopic info /rrbot/camera1/image_raw
rostopic info /isHit

# 4. Monitor topics
rostopic echo /cmd_vel           # Velocity commands
rostopic echo /isHit             # Collision detection
rostopic hz /rrbot/camera1/image_raw  # Camera rate

# 5. Check Gazebo plugins loaded
rosparam list | grep gazebo

# 6. View all nodes
rosnode list
```

---

## What Changed

I fixed the `robbie.xacro` file to include a proper `base_link`:

```xml
<!-- Added base_link for controller -->
<link name="base_link"/>

<joint name="base_to_chassis" type="fixed">
  <parent link="base_link"/>
  <child link="chassis"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

This is required by the `skid_steer_drive_controller` plugin which references `robotBaseFrame` as `base_link`.

**You MUST rebuild** the workspace for this change to take effect:
```bash
cd ~/ENPH-353-COMPETITION
catkin_make
```

---

## Quick Test After Fix

```bash
# 1. Rebuild
cd ~/ENPH-353-COMPETITION && catkin_make && source devel/setup.bash

# 2. Start simulation (if not running)
cd src/enph353/enph353_utils/scripts && ./run_sim.sh -vpgw

# 3. Test movement (new terminal)
cd ~/ENPH-353-COMPETITION/src/robbie/scripts
source ~/ENPH-353-COMPETITION/devel/setup.bash
./test_movement.py
```

Robot should now move! ✓

---

## Still Having Issues?

1. **Check Gazebo console** - Look for error messages
2. **Check ROS logs**: `cat ~/.ros/log/latest/roslaunch-*.log`
3. **Verify URDF loads**: `rosrun xacro xacro $(rospack find robbie)/urdf/robbie.xacro`
4. **Test in RViz**: `rosrun rviz rviz` (add RobotModel display)

---

## Contact/Help

If the robot still won't move after following this guide:
1. Check the exact error messages in terminal
2. Verify Gazebo version: `gazebo --version`
3. Verify ROS version: `rosversion -d`
4. Check the `robbie.xacro` file has the `base_link` addition

**Most common fix**: Rebuild the workspace after URDF changes!
```bash
cd ~/ENPH-353-COMPETITION
catkin_make
source devel/setup.bash
```
