cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_oeq_jitscript_load)

find_package(Torch REQUIRED)
find_package(PythonLibs 3 REQUIRED)

add_executable(load_jitscript load_jitscript.cpp)
set_target_properties(load_jitscript PROPERTIES IMPORTED_LOCATION "${OEQ_EXTLIB}")

target_link_libraries(load_jitscript "${PYTHON_LIBRARIES}")
target_link_libraries(load_jitscript "${TORCH_LIBRARIES}")
#target_link_libraries(load_jitscript -Wl,--no-as-needed "${OEQ_EXTLIB}" "/global/common/software/m1982/vbharadw/conda/my_env/lib/libpython3.so")
target_link_libraries(load_jitscript -Wl,--no-as-needed "${OEQ_EXTLIB}") 
set_property(TARGET load_jitscript PROPERTY CXX_STANDARD 17)