cmake_minimum_required(VERSION 3.17)
project(TP2 C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP)

#add_executable(TP2 main.c)
#add_executable(TP2 aggregation-openmp.c)
#add_executable(TP2 mapreduce-openmp.c)
add_executable(TP2 traffic-openmp.c)

if(OpenMP_CXX_FOUND)
    target_link_libraries(TP2 PUBLIC OpenMP::OpenMP_CXX)
endif()