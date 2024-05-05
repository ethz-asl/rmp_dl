# TUNNEL BINDINGS
add_library(tunnel SHARED
        src/bindings/nvblox_tunnel.cc)
add_dependencies(tunnel ${catkin_EXPORTED_TARGETS})
target_link_libraries(tunnel PRIVATE 
        nvblox::nvblox_lib
        ${catkin_LIBRARIES}
)

# NVBLOX SCENE BINDINGS
set(LIBRARY_NAME nvbloxSceneBindings)
add_library(${LIBRARY_NAME} SHARED
        src/bindings/nvblox_scene_bindings.cc
        )

target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)

add_dependencies(${LIBRARY_NAME} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${LIBRARY_NAME} PRIVATE
        ${TORCH_LIBRARIES}
        ${LIBTORCH_PYTHON}
        Python::Python
        nvblox::nvblox_lib
        tunnel

        # from: https://github.com/ipab-slmc/pybind11_catkin/blob/master/cmake/pybind11_catkin.cmake.in
        ${PYTHON_LIBRARIES}
        ${catkin_LIBRARIES}
        )
target_include_directories(${LIBRARY_NAME} PUBLIC ${catkin_INCLUDE_DIRS})

set_target_properties(${LIBRARY_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}
        PREFIX "") # We add an empty prefix here to make sure it does not add `lib` to every .so file. This way we can properly import in python


install(TARGETS ${LIBRARY_NAME}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )
