cmake_minimum_required(VERSION 3.6)
project(digit_recognizer)

add_executable(digit_recognizer main.cc net.hh meta_helpers.hh array_operations.hh)
target_compile_options(digit_recognizer PUBLIC -O3 -Wall -Wextra -std=c++1z)
