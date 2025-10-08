cmake_minimum_required(VERSION 3.22)
project(xgboostc++ LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-15")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(
    -Wall 
    -Wextra 
    -Wpedantic 
    -Wshadow 
    -O3 
    -fdiagnostics-color=always
)
file(GLOB SOURCES "*.cpp")

file(GLOB HEADERS "*.h")

add_executable(xgb_mini ${SOURCES} ${HEADERS})

target_link_libraries(xgb_mini PRIVATE pthread dl)