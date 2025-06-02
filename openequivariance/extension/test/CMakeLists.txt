cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(load_jitscript load_jitscript.cpp)
target_link_libraries(load_jitscript "${TORCH_LIBRARIES}")
set_property(TARGET load_jitscript PROPERTY CXX_STANDARD 17)