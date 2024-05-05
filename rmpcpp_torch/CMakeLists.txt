cmake_minimum_required(VERSION 3.24)
project(rmpcpp_torch LANGUAGES CUDA CXX)

find_package(nvblox REQUIRED)
find_package(catkin REQUIRED COMPONENTS 
  rmpcpp_planner
  )
  
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS " -Wall -Wextra")

find_package(CUDAToolkit REQUIRED) 
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CMAKE_CUDA_ARCHITECTURES 61)

# Append the torch libraries to cmake prefix path
exec_program("python -c \'import torch;print\(torch.utils.cmake_prefix_path\)\'"
  OUTPUT_VARIABLE torch_path)
list(APPEND CMAKE_PREFIX_PATH ${torch_path})

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch 1.9.1 EXACT REQUIRED)
#### I don't quite understand why, but just doing find_package does not find the libtorch_python.so library, so we find it manually and explicitly link
# against it later. 
set(LIBTORCH_PYTHON "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so")

catkin_package( 
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
    rmpcpp_planner 
)

# Only set these if you want to debug cuda things as well. Makes things incredibly slow otherwise
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

#####################################
#### COPIED FROM NVBLOX:
# Suppress spammy Eigen CUDA warnings.
# "expt-relaxed-constexpr" allows sharing constexpr between host and device
# code.
# "display_error_number" shows a warning number with all warnings, and the
# rest is just suppressing specific warnings from Eigen. Note that the numbers
# keep changing with every CUDA release so this list is a bit arbitrary.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr  --generate-line-info -lineinfo -Xcudafe --display_error_number -Xcudafe --diag_suppress=2977  -Xcudafe --diag_suppress=3057  -Xcudafe --diag_suppress=3059 ")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC -arch=sm_61")
MESSAGE(WARN ${CUDA_NVCC_FLAGS})
# New warning numbers above CUDA 11.2.
if (CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 2)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=20012 -Xcudafe --diag_suppress=20011 -Xcudafe --diag_suppress=20014")
endif ()
######################################
find_package(CUDA REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


include_directories(
  include 
 ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
 ${catkin_INCLUDE_DIRS}
)

include(cmake/nvbloxLayerBindings.cmake)
include(cmake/worldgen.cmake)
include(cmake/nvbloxConversions.cmake)
include(cmake/rmpPlannerBindings.cmake)
include(cmake/policyBindings.cmake)
include(cmake/kernelBindings.cmake)
include(cmake/nvbloxSceneBindings.cmake)
include(cmake/fuserBindings.cmake)
include(cmake/nvbloxSerializerBindings.cmake)

# Comparison planners
include(cmake/chomp.cmake)
include(cmake/rrt.cmake)

# include(cmake/rrt_debug.cmake)

catkin_install_python(PROGRAMS
  #Scripts go here
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})