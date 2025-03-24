cmake_minimum_required(VERSION 2.8.3)
project(parfilter_decision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  visualization_msgs
  olfaction_msgs
  message_generation
  tf
  geometry_msgs
)


catkin_package(
  CATKIN_DEPENDS rospy std_msgs visualization_msgs olfaction_msgs message_runtime tf geometry_msgs
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

