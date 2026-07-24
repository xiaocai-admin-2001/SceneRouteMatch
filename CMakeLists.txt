cmake_minimum_required(VERSION 3.16)
project(road_segment_test_service LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBHV_SOURCE_DIR "/home/cpk/src/libhv-1.3.3" CACHE PATH "libhv source directory")
set(OPENCV_DIR "/home/cpk/local/opencv/lib/cmake/opencv4" CACHE PATH "OpenCVConfig.cmake directory")

set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
set(WITH_HTTP ON CACHE BOOL "" FORCE)
set(WITH_HTTP_SERVER ON CACHE BOOL "" FORCE)
set(WITH_HTTP_CLIENT ON CACHE BOOL "" FORCE)
set(WITH_OPENSSL OFF CACHE BOOL "" FORCE)
set(WITH_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory(${LIBHV_SOURCE_DIR} ${CMAKE_BINARY_DIR}/libhv-build)

find_package(OpenCV REQUIRED CONFIG PATHS ${OPENCV_DIR} NO_DEFAULT_PATH)

add_executable(road_segment_test_service src/main.cpp)
target_include_directories(road_segment_test_service PRIVATE
    ${LIBHV_SOURCE_DIR}
    ${LIBHV_SOURCE_DIR}/base
    ${LIBHV_SOURCE_DIR}/cpputil
    ${LIBHV_SOURCE_DIR}/event
    ${LIBHV_SOURCE_DIR}/ssl
    ${LIBHV_SOURCE_DIR}/evpp
    ${LIBHV_SOURCE_DIR}/http
    ${LIBHV_SOURCE_DIR}/http/server
    ${LIBHV_SOURCE_DIR}/http/client
    ${LIBHV_SOURCE_DIR}/include
)
target_link_libraries(road_segment_test_service PRIVATE hv_static ${OpenCV_LIBS} pthread)
set_target_properties(road_segment_test_service PROPERTIES
    BUILD_RPATH "\$ORIGIN/lib;/home/cpk/local/opencv/lib"
    INSTALL_RPATH "\$ORIGIN/lib"
)


