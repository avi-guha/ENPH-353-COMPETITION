# 2025 Fizz Detective competition [IN PROGRESS]

The repository contains the following ROS packages:

| Folders         | Description      |
|:--------------- |:---------------- |
| enph353_gazebo  | describes simulation world |
| enph353_npcs    | describes and controls pedestrian and Ford truck |
| enph353_utils   | contains competition startup scripts |
| adeept_awr      | describes and controls simulated Adeept AWR robot |
| adeept_awr_ros_driver | controls real world Adeept AWR robot |

## Installation instructions:
** Prerequisites: Ubuntu 20.04 with ROS Noetic installed **

* If you **do not** have a workspace already create one in your home directory.
```
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
```

* Clone the repository into a catkin workspace src folder.
```
git clone https://github.com/ENPH353/2024_competition.git
```

* Build the packages
```
cd ~/ros_ws
catkin_make
```

## Starting the competition:

* Source the environment
```
source devel/setup.bash
```

* Start the simulated world
```
cd src/2024_competition/enph353/enph353_utils/scripts
./run_sim.sh -vpgw
```
The available options are:

| Option | Description      |
|:-------|:---------------- |
| -g     | generate new plates |
| -p     | spawn pedestrian |
| -v     | spawn vehicle    |
| -w     | wind blowing     |

* Start the score tracking app
Open a new tab in the current terminal window by pressing Ctrl+Shift+T 
The new terminal should already be in:
```
~/ros_ws/src/2025_competition/enph353/enph353_utils/scripts
```
Launch the score tracking app:
```
./score_tracker.py
```

## Shutting down the competition:

I suggest you create an alias in your .bashrc file to terminate all the 
processes related to Gazebo and ROS. Adding something like below should 
be sufficient:
```
alias killjim="pkill -f gzserver; pkill -f gzclient; killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient"
```
Then when you need to stop the simulation call 
```
killjim
```
from a new terminal.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
    <img alt="Creative Commons Licence" style="border-width:0" 
        src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
</a><br />
This work is licensed under a 
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
    Creative Commons Attribution-ShareAlike 4.0 International License</a>.
