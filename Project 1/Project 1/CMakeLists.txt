cmake_minimum_required(VERSION 3.24)
project(Project_1)

set(CMAKE_CXX_STANDARD 14)

include_directories(GLES2)

add_executable(Project_1
        GLES2/gl2.h
        GLES2/gl2ext.h
        GLES2/gl2platform.h
        nn.cpp)
