cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)
set(CAMKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" -std=c++1y)
project(Image_Processing)


file(GLOB source_files "*.cpp" header_files "*.h")
#target_link_libraries(${PROJECT_NAME} PRIVATE "D:/opencv/build/x64/vc15/lib")

add_executable(${PROJECT_NAME} ${source_files} ${header_files})