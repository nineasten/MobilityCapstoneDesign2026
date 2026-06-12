cmake_minimum_required(VERSION 3.0.2)
project(destination_selector)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/destination_selector_node.py
  scripts/voxel_map_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
