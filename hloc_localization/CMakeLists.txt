cmake_minimum_required(VERSION 3.0.2)
project(hloc_localization)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_srvs
  std_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/hloc_localization_node.py
  scripts/hloc_pose_marker_node.py
  scripts/hloc_persistent_localization_node.py
  scripts/hloc_reference_pointcloud_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
