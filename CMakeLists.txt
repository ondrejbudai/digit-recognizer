cmake_minimum_required(VERSION 3.9)
project(digit_recognizer)

set(CMAKE_CXX_STANDARD 17)

add_executable(digit_recognizer main.cc net.hh meta_helpers.hh)
target_compile_options(digit_recognizer PUBLIC -O3 -Wall -Wextra)
