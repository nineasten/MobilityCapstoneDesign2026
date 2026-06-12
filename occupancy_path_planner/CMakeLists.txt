cmake_minimum_required(VERSION 3.0.2)
project(occupancy_path_planner)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/occupancy_path_planner_node.py
  scripts/path_image_overlay.py
  scripts/path_live_image_overlay.py
  scripts/unity_path_processor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
