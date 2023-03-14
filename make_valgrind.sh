#!/bin/bash

mkdir -p build/debug/
cmake -B ./build/debug/ -DCMAKE_BUILD_TYPE=Debug -DSTATIC_LIB=OFF -DTEST_COVERAGE=OFF -DUB_SANITIZER=OFF
cd ./build/debug/ || exit
make clean
make idealnn
make idealnn_test
ctest --verbose -C valgrind
