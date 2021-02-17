add_subdirectory(./../lib shared_lib)

include_directories(${SHARED_LIB_INCLUDE_DIR})

set(SRCS gameoflife_Seq.c)
add_executable(TP4_Seq ${SRCS})
target_link_libraries(TP4_Seq shared_lib)
