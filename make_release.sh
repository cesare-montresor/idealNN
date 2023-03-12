#!/bin/bash

cmake -B ./build/release/ -DCMAKE_BUILD_TYPE=Release -DTEST_COVERAGE=OFF -DTEST_COVERAGE=OFF -DUB_SANITIZER=OFF
cd ./build/release/ || exit
make clean
make idealnn
make doc
make install