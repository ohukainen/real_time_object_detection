cmake_minimum_required(VERSION 3.20)

project(real_time_object_detection)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(${PROJECT_NAME} 
    src/main.cpp
    
    src/Input.cpp
    src/ModelYOLO.cpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} opencv::opencv)
target_link_libraries(${PROJECT_NAME} nlohmann_json::nlohmann_json)
