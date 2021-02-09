cmake_minimum_required(VERSION 3.17)
project(TP3 C)

set(CMAKE_C_STANDARD 99)
include_directories(SYSTEM ${MSMPI_INC})

find_package(MPI REQUIRED)

#add_executable(TP3 main.c)
add_executable(TP3 text2digits.c)


if(MPI_C_FOUND)
    target_link_libraries(TP3 ${MSMPI_LIB64})
endif()