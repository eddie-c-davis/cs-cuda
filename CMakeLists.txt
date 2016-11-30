cmake_minimum_required(VERSION 3.6)
project(trunk)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    Makefile
    mandelbrot.c
    timer.c
    timer.h)

add_executable(trunk ${SOURCE_FILES})