cmake_minimum_required(VERSION 3.16)
project(TP4 C)
set(CMAKE_C_STANDARD 99)

add_subdirectory(src/lib)

add_subdirectory(src/seq)
add_subdirectory(src/mpi)