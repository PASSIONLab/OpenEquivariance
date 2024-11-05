rm -rf build; mkdir build;
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PWD} \
        -DPYTHON_EXECUTABLE=$(which python)
popd
