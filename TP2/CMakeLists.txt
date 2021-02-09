cmake_minimum_required(VERSION 3.17)
project(TP2 C)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP)


#add_executable(TP2 main.c)
#add_executable(TP2 aggregation-openmp.c)
#add_executable(TP2 mapreduce-openmp.c)
#add_executable(TP2 traffic-openmp.c)
add_executable(TP2 philosophers-openmp.c)

if(OpenMP_C_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    target_link_libraries(TP2 PUBLIC OpenMP::OpenMP_C)
endif()