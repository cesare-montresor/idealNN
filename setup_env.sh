#!/bin/bash

sudo apt update
sudo apt install -y libeigen3-dev
sudo apt install -y valgrind gcovr cloc
sudo apt install -y doxygen
sudo apt install -y texlive texlive-font-utils
sudo apt install -y graphviz

mkdir -p build/debug/
mkdir -p build/release/