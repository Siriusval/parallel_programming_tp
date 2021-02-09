cmake_minimum_required(VERSION 3.17)
project(TP3 C)

set(CMAKE_C_STANDARD 99)
find_package(MPI REQUIRED)

#add_executable(TP3 main.c)
#add_executable(TP3 text2digits.c)
add_executable(TP3 text2digitsMPI.c)
target_link_libraries(TP3 PUBLIC MPI::MPI_C)

