cmake_minimum_required(VERSION 3.0.2)
project(turtlebot3_dynamic_pedestrians)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
)

catkin_package()

install(DIRECTORY launch models worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  scripts/scan_visible_pedestrians.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
