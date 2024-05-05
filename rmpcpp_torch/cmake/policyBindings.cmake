set(LIBRARY_NAME policyBindings)
add_library(${LIBRARY_NAME} SHARED
        src/bindings/policy_bindings.cc
        )
target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)

add_dependencies(${LIBRARY_NAME} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${LIBRARY_NAME} PUBLIC ${TORCH_LIBRARIES}
        ${LIBTORCH_PYTHON}
        Python::Python
        nvblox::nvblox_lib

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
