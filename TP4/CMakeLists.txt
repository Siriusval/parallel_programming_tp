cmake_minimum_required(VERSION 3.17)
project(TP4 C)

set(CMAKE_C_STANDARD 99)
find_package(MPI REQUIRED)

#add_executable(TP4 main.c)
#add_executable(TP4 gameoflife.c gameoflife.h)
add_executable(TP4 gameoflife_MPI.c gameoflife.c gameoflife.h)

if(WIN32)
    target_link_libraries(TP4 PUBLIC MPI::MPI_C wsock32 ws2_32)
else()
    target_link_libraries(TP4 PUBLIC MPI::MPI_C)
endif()
