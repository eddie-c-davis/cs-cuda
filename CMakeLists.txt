cmake_minimum_required(VERSION 3.6)
project(trunk)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    Makefile
    mandelbrot.cu)

include_directories(/usr/include /usr/local/include /usr/local/cuda/include)

add_executable(trunk ${SOURCE_FILES})
