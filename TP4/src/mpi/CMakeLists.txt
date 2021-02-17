add_subdirectory(./../lib shared_lib)

include_directories(${SHARED_LIB_INCLUDE_DIR})

set(SRCS gameoflife_MPI.c)
add_executable(TP4_MPI ${SRCS})

find_package(MPI REQUIRED)

if(WIN32)
    target_link_libraries(TP4_MPI PUBLIC shared_lib MPI::MPI_C wsock32 ws2_32)
else()
    target_link_libraries(TP4_MPI PUBLIC shared_lib MPI::MPI_C)
endif()
