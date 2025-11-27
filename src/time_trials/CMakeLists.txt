cmake_minimum_required(VERSION 3.0.2)
project(time_trials)

find_package(catkin REQUIRED)

catkin_package()

## Install Python scripts
catkin_install_python(PROGRAMS
  src/teleop_ps4.py
  src/LineFollowing/camera_viewer.py
  src/LineFollowing/data_collector.py
  src/LineFollowing/inference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
