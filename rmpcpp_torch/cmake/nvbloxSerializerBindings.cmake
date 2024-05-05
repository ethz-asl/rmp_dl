# C string wrapper such that we hide the std::string symbols
add_library(serializer_functions_c_str_wrapper SHARED
        src/bindings/nvblox_serializer_c_str_wrapper.cc
        )

target_compile_features(serializer_functions_c_str_wrapper PRIVATE cxx_std_17)

add_dependencies(serializer_functions_c_str_wrapper ${catkin_EXPORTED_TARGETS})

target_link_libraries(serializer_functions_c_str_wrapper PRIVATE
        nvblox::nvblox_lib
        nvblox::nvblox_datasets
        ${catkin_LIBRARIES}
)
target_compile_options(serializer_functions_c_str_wrapper BEFORE PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)


set(LIBRARY_NAME nvbloxSerializerBindings)
add_library(${LIBRARY_NAME} SHARED
        src/bindings/nvblox_serializer_bindings.cc
        )
target_compile_features(${LIBRARY_NAME} PRIVATE cxx_std_17)

target_link_libraries(${LIBRARY_NAME} PUBLIC 
        ${TORCH_LIBRARIES}
        ${LIBTORCH_PYTHON}
        Python::Python
        nvblox::nvblox_lib
        serializer_functions_c_str_wrapper

        # from: https://github.com/ipab-slmc/pybind11_catkin/blob/master/cmake/pybind11_catkin.cmake.in
        ${PYTHON_LIBRARIES}
        ${catkin_LIBRARIES}
        )
target_include_directories(${LIBRARY_NAME} PUBLIC ${catkin_INCLUDE_DIRS})

set_target_properties(${LIBRARY_NAME} PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}
        PREFIX "") # We add an empty prefix here to make sure it does not add `lib` to every .so file. This way we can properly import in python

install(TARGETS ${LIBRARY_NAME} cstring_wrapper
        LIBRARY DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )


