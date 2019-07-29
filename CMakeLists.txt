cmake_minimum_required(VERSION 2.8.3)
project(baxter_whill_movement)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package(CATKIN_DEPENDS)

include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})
