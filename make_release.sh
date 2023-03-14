#!/bin/bash

mkdir -p build/release/
cmake -B ./build/release/ -DCMAKE_BUILD_TYPE=Release -DSTATIC_LIB=ON -DTEST_COVERAGE=OFF -DUB_SANITIZER=OFF
cd ./build/release/ || exit
make clean
make idealnn
make doc
make install