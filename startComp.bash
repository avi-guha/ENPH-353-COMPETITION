source devel/setup.bash
cd src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw &
sleep 5
cd /home/fizzer/ENPH-353-COMPETITION/src/time_trials/src/LineFollowing
python3 ImageProcessing.py
