cmake_minimum_required(VERSION 3.17)
project(TP4 C)

set(CMAKE_C_STANDARD 99)
find_package(MPI REQUIRED)

#add_executable(TP4 main.c)
add_executable(TP4 gameoflife.c)
target_link_libraries(TP4 PUBLIC MPI::MPI_C)