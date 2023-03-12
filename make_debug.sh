#!/bin/bash

cmake -B ./build/debug/ -DCMAKE_BUILD_TYPE=Debug -DSTATIC_LIB=OFF -DTEST_COVERAGE=ON -DUB_SANITIZER=ON
cd ./build/debug/ || exit
make clean
make idealnn
make idealnn_test
make coverage
make doc