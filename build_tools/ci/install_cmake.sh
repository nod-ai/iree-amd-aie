#!/bin/bash
sudo apt-get update
sudo apt-get install -y bc wget

version=3.23
build=2
limit=3.20

result=$(echo "$version >= $limit" | bc -l)
os=$([ "$result" == 1 ] && echo "linux" || echo "Linux")
mkdir -p ~/temp
cd ~/temp

wget https://cmake.org/files/v$version/cmake-$version.$build-$os-x86_64.sh
sudo mkdir -p /opt/cmake
sudo sh cmake-$version.$build-$os-x86_64.sh --prefix=/opt/cmake --skip-license
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

rm -rf ~/temp
