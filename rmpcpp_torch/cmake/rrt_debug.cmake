set(LIBRARY_NAME rrt_debug)

find_package(ompl REQUIRED)

message(WARNING "Ompl found: ${OMPL_FOUND}\n")
message(WARNING "Ompl include dirs: ${OMPL_INCLUDE_DIRS}\n")

add_executable(${LIBRARY_NAME} 
        src/comparison_planners/rrt/debug.cc
        )
target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)

target_compile_options(${LIBRARY_NAME} PRIVATE -g -O0)

add_dependencies(${LIBRARY_NAME} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${LIBRARY_NAME} PUBLIC 
        ${catkin_LIBRARIES}
        nvblox::nvblox_lib
        ${OMPL_LIBRARIES}
        )
target_include_directories(${LIBRARY_NAME} PUBLIC 
        ${OMPL_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS})

install(TARGETS ${LIBRARY_NAME}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
