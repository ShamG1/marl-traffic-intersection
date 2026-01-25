cmake_minimum_required(VERSION 3.15)

if(NOT WIN32)
    message(FATAL_ERROR "MARLEnv Renderer is Windows-only. Configure/build on Windows.")
endif()

project(MARLEnv LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure Python Development.Module so pybind11NewTools has python3_add_library
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Try pybind11 from Python site-packages first --------------------------------
find_package(pybind11 CONFIG QUIET)

if (NOT pybind11_FOUND)
    # Query python module for CMake dir.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _PY_OK)
    if (_PY_OK EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endif()

if (NOT pybind11_FOUND)
    include(FetchContent)
    set(FETCHCONTENT_QUIET FALSE)
    message(STATUS "Fetching pybind11 via FetchContent...")
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip
        URL_HASH SHA256=4f4aacab5659b264a3ae9935be98b9f96e65e6a85ee661eb5d286276f4e5a381
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
pybind11_add_module(MARLEnv
    bindings.cpp
    Car.cpp
    IntersectionEnv.cpp
    IntersectionEnv_render.cpp
    Lidar.cpp
    RouteGen.cpp
    RoadMask.cpp
    LineMask.cpp
    TrafficFlow.cpp
    Renderer.cpp)

# ----------------------------------------------------------------------------
# OpenGL + GLFW (manual GLFW binary package)
# ----------------------------------------------------------------------------
find_package(OpenGL REQUIRED)

# GLFW binary root; default to TOOLS/glfw-3.4.bin.WIN64 at project root
# You can override from the command line, e.g.:
#   cmake -DGLFW_ROOT=E:/path/to/glfw-3.4.bin.WIN64 ..
if(NOT DEFINED GLFW_ROOT)
    # cpp/ -> MCTS_DUAL/ -> project root (with TOOLS/)
    set(GLFW_ROOT "${CMAKE_SOURCE_DIR}/../TOOLS/glfw-3.4.bin.WIN64")
endif()

set(GLFW_INCLUDE_DIR "${GLFW_ROOT}/include")
# You have lib-vc2022/glfw3.lib, glfw3dll.lib, etc.
set(GLFW_LIB_DIR "${GLFW_ROOT}/lib-vc2022")

if(NOT EXISTS "${GLFW_INCLUDE_DIR}/GLFW/glfw3.h")
    message(FATAL_ERROR "Cannot find GLFW headers in ${GLFW_INCLUDE_DIR}. Check GLFW_ROOT")
endif()

if(EXISTS "${GLFW_LIB_DIR}/glfw3.lib")
    set(GLFW_LIB_NAME glfw3)
elseif(EXISTS "${GLFW_LIB_DIR}/glfw3dll.lib")
    set(GLFW_LIB_NAME glfw3dll)
else()
    message(FATAL_ERROR "Cannot find glfw3.lib or glfw3dll.lib in ${GLFW_LIB_DIR}. Check GLFW_ROOT")
endif()

target_include_directories(MARLEnv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${GLFW_INCLUDE_DIR})
target_link_directories(MARLEnv PRIVATE ${GLFW_LIB_DIR})

# Windows system OpenGL
target_link_libraries(MARLEnv PRIVATE ${GLFW_LIB_NAME} opengl32)

if(WIN32)
    target_link_libraries(MARLEnv PRIVATE gdi32 gdiplus)

    # Avoid MSVC warning C4819 (source file contains characters not representable in current code page)
    target_compile_options(MARLEnv PRIVATE /utf-8)
endif()
