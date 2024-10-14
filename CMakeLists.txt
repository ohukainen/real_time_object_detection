cmake_minimum_required(VERSION 3.20)

project(real_time_object_detection)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV CONFIG REQUIRED)

add_executable(${PROJECT_NAME} 
    src/main.cpp
    
    src/InputCamera.cpp
    src/ModelONNX.cpp
    src/Inference.cpp
)


target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} opencv::opencv)
