cmake_minimum_required(VERSION 3.20)

project(OrbViSLAMBindings LANGUAGES CXX)

# ------------------------------------------------------------
# Basic build settings
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Helpful on Windows / multi-config generators
if (MSVC)
  add_compile_options(/utf-8)
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# ------------------------------------------------------------
# Third-party roots
# ------------------------------------------------------------
set(ORB_SLAM3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/orb_slam3" CACHE PATH "Path to ORB-SLAM3 source tree")
set(OBVI_SLAM_ROOT  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/obvi_slam"  CACHE PATH "Path to ObVi-SLAM source tree")

if (NOT EXISTS "${ORB_SLAM3_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR "ORB-SLAM3 CMakeLists.txt not found at: ${ORB_SLAM3_ROOT}")
endif()

# If ORB-SLAM3 is a normal CMake project, add it as a subdirectory.
# This is the cleanest way to reuse its target(s).
add_subdirectory("${ORB_SLAM3_ROOT}" "${CMAKE_BINARY_DIR}/third_party/orb_slam3-build")

# ------------------------------------------------------------
# Common include paths
# ------------------------------------------------------------
set(COMMON_PROJECT_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${ORB_SLAM3_ROOT}/include"
  "${ORB_SLAM3_ROOT}/Thirdparty/Sophus"
  "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2"
  "${ORB_SLAM3_ROOT}/Thirdparty/g2o"
)

# ------------------------------------------------------------
# 1) ORB-SLAM3 Python bindings
# ------------------------------------------------------------
nanobind_add_module(obvi_orb_slam3
  src/orb_slam3/pybind.cpp
)

target_include_directories(obvi_orb_slam3 PRIVATE
  ${COMMON_PROJECT_INCLUDES}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# ORB-SLAM3 upstream build typically exposes a target named ORB_SLAM3.
# If your local third_party/orb_slam3 uses a different target name,
# replace ORB_SLAM3 below with the actual target name.
target_link_libraries(obvi_orb_slam3 PRIVATE
  ORB_SLAM3
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Threads::Threads
)

# Keep the module name clean on disk if needed
set_target_properties(obvi_orb_slam3 PROPERTIES
  OUTPUT_NAME "obvi_orb_slam3"
)

# ------------------------------------------------------------
# 2) ObVi-SLAM bridge bindings
# ------------------------------------------------------------
nanobind_add_module(obvi_slam_bridge
  src/obvi_slam/bindings.cpp
  src/obvi_slam/bridge_runner.cpp
  src/obvi_slam/third_party_adapter.cpp
)

target_include_directories(obvi_slam_bridge PRIVATE
  ${COMMON_PROJECT_INCLUDES}
  "${OBVI_SLAM_ROOT}"
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# For now the ObVi-SLAM bridge is self-contained and the adapter is a stub.
# Once you wire the real upstream call, you can link any required ObVi-SLAM
# target(s) here, e.g.:
#
# target_link_libraries(obvi_slam_bridge PRIVATE obvi_slam_core ...)
#
# If the adapter later uses ORB-SLAM3 types directly, you may also add:
target_link_libraries(obvi_slam_bridge PRIVATE
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Threads::Threads
  ORB_SLAM3
)

set_target_properties(obvi_slam_bridge PROPERTIES
  OUTPUT_NAME "obvi_slam_bridge"
)

# ------------------------------------------------------------
# Optional install targets
# ------------------------------------------------------------
install(TARGETS obvi_orb_slam3 obvi_slam_bridge
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
  ARCHIVE DESTINATION .
)