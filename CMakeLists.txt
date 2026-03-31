cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

if(POLICY CMP0082)
    cmake_policy(SET CMP0082 NEW)
endif()

# ------------------------------------------------------------------------------
# vcpkg auto-detection
# Must happen before project()
# ------------------------------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(_orbvi_vcpkg_candidates "")

    if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE} AND EXISTS "$ENV{CMAKE_TOOLCHAIN_FILE}")
        list(APPEND _orbvi_vcpkg_candidates "$ENV{CMAKE_TOOLCHAIN_FILE}")
    endif()

    if(DEFINED ENV{VCPKG_ROOT})
        list(APPEND _orbvi_vcpkg_candidates
            "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    endif()

    if(DEFINED ENV{HOME})
        list(APPEND _orbvi_vcpkg_candidates
            "$ENV{HOME}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    endif()

    if(UNIX AND NOT APPLE)
        list(APPEND _orbvi_vcpkg_candidates
            "/mnt/c/vcpkg/scripts/buildsystems/vcpkg.cmake")
    endif()

    foreach(_orbvi_toolchain IN LISTS _orbvi_vcpkg_candidates)
        if(EXISTS "${_orbvi_toolchain}")
            set(CMAKE_TOOLCHAIN_FILE "${_orbvi_toolchain}" CACHE FILEPATH "vcpkg toolchain file")
            message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
            break()
        endif()
    endforeach()

    unset(_orbvi_toolchain)
    unset(_orbvi_vcpkg_candidates)
endif()

project(orbvi_slam VERSION 0.1.0 LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Global build settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PACKAGE_NAME "${PROJECT_NAME}")
set(SRC_ROOT    "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ORB_ROOT    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/orb_slam3")
set(OBVI_ROOT   "${CMAKE_CURRENT_SOURCE_DIR}/third_party/obvi_slam")

# ------------------------------------------------------------------------------
# Third-party subprojects
# ------------------------------------------------------------------------------
set(SUPPRESS_WARNINGS ON CACHE BOOL "Suppress compiler warnings" FORCE)

add_subdirectory(third_party/orb_slam3)

set(USE_ROS OFF CACHE BOOL "Disable ROS support" FORCE)
set(SUPPRESS_WARNINGS ON CACHE BOOL "Suppress compiler warnings" FORCE)
add_subdirectory(third_party/obvi_slam)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres CONFIG REQUIRED)

# SuiteSparse fallback
find_package(SuiteSparse QUIET)
if(SuiteSparse_FOUND)
    set(ORBVI_SUITESPARSE_LIBS ${SuiteSparse_LIBRARIES})
else()
    find_library(CHOLMOD_LIB NAMES cholmod REQUIRED)
    set(ORBVI_SUITESPARSE_LIBS ${CHOLMOD_LIB})
endif()

# ------------------------------------------------------------------------------
# Helper: install internal shared libraries next to the Python modules
# ------------------------------------------------------------------------------
# This is important on Linux when ORB_SLAM3 / ut_vslam are built as shared libs:
# the extension modules can then find them via $ORIGIN at import time.
function(orbvi_install_if_shared target_name)
    if(TARGET ${target_name})
        get_target_property(_type ${target_name} TYPE)
        if(_type STREQUAL "SHARED_LIBRARY" OR _type STREQUAL "MODULE_LIBRARY")
            if(UNIX AND NOT APPLE)
                set_target_properties(${target_name} PROPERTIES
                    INSTALL_RPATH "$ORIGIN"
                )
            endif()

            install(TARGETS ${target_name}
                LIBRARY DESTINATION "${PACKAGE_NAME}"
                RUNTIME DESTINATION "${PACKAGE_NAME}"
            )
        endif()
    endif()
endfunction()

# ------------------------------------------------------------------------------
# ORB-SLAM3 nanobind module
# ------------------------------------------------------------------------------
nanobind_add_module(_orb_slam3
    "${SRC_ROOT}/orb_slam3.cpp"
)

target_link_libraries(_orb_slam3 PRIVATE
    ORB_SLAM3
)

target_include_directories(_orb_slam3 PRIVATE
    "${SRC_ROOT}"
    "${ORB_ROOT}/include"
    "${ORB_ROOT}/include/CameraModels"
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

if(WIN32)
    target_compile_definitions(_orb_slam3 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# ------------------------------------------------------------------------------
# ObVi-SLAM nanobind module
# ------------------------------------------------------------------------------
nanobind_add_module(_obvi_slam
    "${SRC_ROOT}/obvi_slam.cpp"
)

target_link_libraries(_obvi_slam PRIVATE
    ut_vslam
    Ceres::ceres
    ${ORBVI_SUITESPARSE_LIBS}
)

target_include_directories(_obvi_slam PRIVATE
    "${SRC_ROOT}"
    "${OBVI_ROOT}/include"
    "${OBVI_ROOT}/src"
    "${OBVI_ROOT}/src/shared"
    "${ORB_ROOT}/include"
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(_obvi_slam PRIVATE
    CONSTRAIN_ELLIPSOID_ORIENTATION
    _USE_MATH_DEFINES
)

if(WIN32)
    target_compile_definitions(_obvi_slam PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# On Linux, prefer sibling shared libraries in the installed package.
if(UNIX AND NOT APPLE)
    set_target_properties(_orb_slam3 PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
    set_target_properties(_obvi_slam PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# ------------------------------------------------------------------------------
# Install binary extension modules first
# ------------------------------------------------------------------------------
install(TARGETS _orb_slam3 _obvi_slam
    LIBRARY DESTINATION "${PACKAGE_NAME}"
    RUNTIME DESTINATION "${PACKAGE_NAME}"
)

# Install internal shared libraries next to the modules when applicable.
orbvi_install_if_shared(ORB_SLAM3)
orbvi_install_if_shared(ut_vslam)

# ------------------------------------------------------------------------------
# Install-time runtime dependency bundling
# ------------------------------------------------------------------------------
# Windows: copy resolved DLL dependencies next to the installed .pyd files.
# We avoid install(TARGETS ... RUNTIME_DEPENDENCIES ...) because unresolved
# system DLLs can fail the wheel build.
if(WIN32 AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    install(CODE "
        set(_pkgdir \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${PACKAGE_NAME}\")

        file(GLOB _installed_modules
            \"\${_pkgdir}/_orb_slam3*.pyd\"
            \"\${_pkgdir}/_obvi_slam*.pyd\"
        )

        if(NOT _installed_modules)
            message(FATAL_ERROR \"Installed nanobind modules were not found in: \${_pkgdir}\")
        endif()

        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR _resolved
            UNRESOLVED_DEPENDENCIES_VAR _unresolved
            MODULES \${_installed_modules}
            DIRECTORIES
                \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin\"
                \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin\"
            PRE_EXCLUDE_REGEXES
                \"^api-ms-.*\"
                \"^ext-ms-.*\"
                \"^hvsifiletrust\\.dll$\"
                \"^pdmutilities\\.dll$\"
                \"^python3[0-9]+(_d)?\\.dll$\"
            POST_EXCLUDE_REGEXES
                \".*system32/.*\\\\.dll$\"
                \".*windows/.*\\\\.dll$\"
        )

        set(_unexpected_unresolved)
        foreach(_dep IN LISTS _unresolved)
            string(TOLOWER \"\${_dep}\" _dep_lc)
            if(NOT _dep_lc MATCHES \"^api-ms-.*\"
               AND NOT _dep_lc MATCHES \"^ext-ms-.*\"
               AND NOT _dep_lc MATCHES \"^hvsifiletrust\\.dll$\"
               AND NOT _dep_lc MATCHES \"^pdmutilities\\.dll$\"
               AND NOT _dep_lc MATCHES \"^python3[0-9]+(_d)?\\.dll$\")
                list(APPEND _unexpected_unresolved \"\${_dep}\")
            endif()
        endforeach()

        if(_unexpected_unresolved)
            message(FATAL_ERROR
                \"Unexpected unresolved runtime dependencies: \${_unexpected_unresolved}\")
        endif()

        if(_resolved)
            list(REMOVE_DUPLICATES _resolved)
            file(INSTALL
                DESTINATION \"\${_pkgdir}\"
                TYPE FILE
                FILES \${_resolved}
            )
        endif()
    ")
elseif(UNIX AND NOT APPLE AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    # Linux + vcpkg: optionally copy non-system runtime deps when available.
    install(CODE "
        set(_pkgdir \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${PACKAGE_NAME}\")

        file(GLOB _installed_modules
            \"\${_pkgdir}/_orb_slam3*.so\"
            \"\${_pkgdir}/_obvi_slam*.so\"
        )

        if(_installed_modules)
            file(GET_RUNTIME_DEPENDENCIES
                RESOLVED_DEPENDENCIES_VAR _resolved
                UNRESOLVED_DEPENDENCIES_VAR _unresolved
                MODULES \${_installed_modules}
                DIRECTORIES
                    \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib\"
                    \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib\"
                POST_EXCLUDE_REGEXES
                    \"^/lib/.*\"
                    \"^/lib64/.*\"
                    \"^/usr/lib/.*\"
                    \"^/usr/lib64/.*\"
                    \"^/usr/local/lib/.*\"
            )

            if(_unresolved)
                message(FATAL_ERROR
                    \"Unexpected unresolved runtime dependencies: \${_unresolved}\")
            endif()

            if(_resolved)
                list(REMOVE_DUPLICATES _resolved)
                file(INSTALL
                    DESTINATION \"\${_pkgdir}\"
                    TYPE FILE
                    FILES \${_resolved}
                )
            endif()
        endif()
    ")
endif()

# ------------------------------------------------------------------------------
# Generate stubs via a Python wrapper executed in-process
# ------------------------------------------------------------------------------
# Why not nanobind_add_stub() directly?
# On Windows / Python 3.8+, importing a .pyd may require os.add_dll_directory()
# for dependent native libraries. Also, a temporary __init__.py in the staged
# package prevents the source tree package from winning via namespace merging.

set(STUBGEN_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/run_nanobind_stubgen.py")

file(WRITE "${STUBGEN_WRAPPER}" [=[
import os
import sys
from pathlib import Path

_DLL_DIR_HANDLES = []

def add_dll_dir(path: str) -> None:
    if os.name == "nt" and hasattr(os, "add_dll_directory") and path and os.path.isdir(path):
        _DLL_DIR_HANDLES.append(os.add_dll_directory(path))

def main() -> int:
    install_prefix = Path(sys.argv[1]).resolve()
    package_name = sys.argv[2]
    source_root = Path(sys.argv[3]).resolve()
    vcpkg_bin = sys.argv[4] if len(sys.argv) > 4 else ""
    vcpkg_debug_bin = sys.argv[5] if len(sys.argv) > 5 else ""

    package_dir = install_prefix / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    # Make the staged package a regular package so it wins over the source tree.
    temp_init = package_dir / "__init__.py"
    temp_init.write_text("# temporary stubgen package marker\n", encoding="utf-8")

    # Prevent accidental imports from the source tree.
    os.chdir(str(install_prefix))

    cleaned_sys_path = []
    for p in sys.path:
        try:
            resolved = Path(p).resolve() if p else None
        except Exception:
            cleaned_sys_path.append(p)
            continue

        if resolved is not None and resolved == source_root:
            continue

        cleaned_sys_path.append(p)

    # Ensure the wheel staging root is first.
    sys.path[:] = [str(install_prefix)] + cleaned_sys_path

    # Windows DLL search paths for dependent native libraries.
    add_dll_dir(str(package_dir))
    add_dll_dir(vcpkg_bin)
    add_dll_dir(vcpkg_debug_bin)

    import nanobind.stubgen as stubgen

    sys.argv = [
        "nanobind.stubgen",
        "-i", str(install_prefix),
        "-m", f"{package_name}._orb_slam3",
        "-m", f"{package_name}._obvi_slam",
        "-O", str(package_dir),
        "-M", str(package_dir / "py.typed"),
    ]

    stubgen.main()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
]=])

# ------------------------------------------------------------------------------
# Generate a separate install script to avoid fragile quoting in install(CODE)
# ------------------------------------------------------------------------------
file(TO_CMAKE_PATH "${Python_EXECUTABLE}" PYTHON_EXECUTABLE_CMAKE)
file(TO_CMAKE_PATH "${STUBGEN_WRAPPER}" STUBGEN_WRAPPER_CMAKE)
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" SOURCE_ROOT_CMAKE)

if(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    file(TO_CMAKE_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin" VCPKG_BIN_CMAKE)
    file(TO_CMAKE_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin" VCPKG_DEBUG_BIN_CMAKE)
else()
    set(VCPKG_BIN_CMAKE "")
    set(VCPKG_DEBUG_BIN_CMAKE "")
endif()

set(STUBGEN_INSTALL_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/run_stubgen_install.cmake")

set(_stubgen_install_script_template [=[
set(_install_prefix "$ENV{DESTDIR}@CMAKE_INSTALL_PREFIX@")

execute_process(
    COMMAND
        "@PYTHON_EXECUTABLE_CMAKE@"
        "@STUBGEN_WRAPPER_CMAKE@"
        "${_install_prefix}"
        "@PACKAGE_NAME@"
        "@SOURCE_ROOT_CMAKE@"
        "@VCPKG_BIN_CMAKE@"
        "@VCPKG_DEBUG_BIN_CMAKE@"
    RESULT_VARIABLE _stubgen_result
)

if(NOT _stubgen_result EQUAL 0)
    message(FATAL_ERROR "nanobind stub generation failed with exit code ${_stubgen_result}")
endif()
]=])

string(CONFIGURE "${_stubgen_install_script_template}" _stubgen_install_script @ONLY)
file(WRITE "${STUBGEN_INSTALL_SCRIPT}" "${_stubgen_install_script}")

install(SCRIPT "${STUBGEN_INSTALL_SCRIPT}")

# ------------------------------------------------------------------------------
# Install the real __init__.py last, after stub generation
# ------------------------------------------------------------------------------
install(FILES
    "${SRC_ROOT}/__init__.py"
    DESTINATION "${PACKAGE_NAME}"
)