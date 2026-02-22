cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(test_oeq_aoti_load)

find_package(Torch REQUIRED)

add_executable(load_aoti load_aoti.cpp)
target_link_libraries(load_aoti "${TORCH_LIBRARIES}")
target_link_libraries(load_aoti -Wl,--no-as-needed "${OEQ_EXTLIB}") 
set_property(TARGET load_aoti PROPERTY CXX_STANDARD 17)
