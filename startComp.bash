cd /home/fizzer/ENPH-353-COMPETITION
source devel/setup.bash

cd src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw &
sleep 3

cd /home/fizzer/ENPH-353-COMPETITION
source devel/setup.bash

# Start the line following inference node in background
python3 src/competition/line_following/inference_node.py &
sleep 1

# Start the clueboard detector in background
python3 src/competition/clueboard_detection/board_detector.py &
sleep 1

# Start the score tracker (foreground - GUI)
cd src/enph353/enph353_utils/scripts
./score_tracker.py

