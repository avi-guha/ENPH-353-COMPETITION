source devel/setup.bash
cd src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw &
sleep 5

# Launch PS4 Teleop (and disable default controller via launch file)
roslaunch time_trials time_trials_launch.launch &
sleep 3

# Launch inference node
cd /home/fizzer/ENPH-353-COMPETITION/src/time_trials/src/LineFollowing
# python3 inference_node.py

#Launch data collector 
python3 data_collector.py
python3 camera_viewer.py