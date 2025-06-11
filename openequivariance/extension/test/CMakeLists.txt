cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(test_oeq_jitscript_load)

find_package(Torch REQUIRED)

add_executable(load_jitscript load_jitscript.cpp)
target_link_libraries(load_jitscript "${TORCH_LIBRARIES}")
target_link_libraries(load_jitscript -Wl,--no-as-needed "${OEQ_EXTLIB}") 
set_property(TARGET load_jitscript PROPERTY CXX_STANDARD 17)