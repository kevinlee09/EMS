cmake_minimum_required(VERSION 3.6)

project(orient2d)

set(SOURCE_FILES Im.hpp OrientMap.hpp Vec.hpp Color.h
       util.h main.cpp)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIR})

FIND_PACKAGE(PNG  REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

#添加链接库
TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
                      ${OpenCV_LIBS} 
                      ${JPEG_LIBRARIES}
                      ${PNG_LIBRARIES}
                      fftw3
                      -pthread -O3 -fopenmp)