cd /home/fizzer/ENPH-353-COMPETITION
source devel/setup.bash

cd src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw &
sleep 3

cd /home/fizzer/ENPH-353-COMPETITION
source devel/setup.bash

cd src/enph353/enph353_utils/scripts
./score_tracker.py

