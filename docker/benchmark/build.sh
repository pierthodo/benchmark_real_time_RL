#!/usr/bin/env bash

# usage: ./build.sh <image-name> <rtrl-revision-hash (optional)>
cd ../base
docker build -t mujoco .
cd ../benchmark

# using experimental ssh forwarding (see Dockerfile)
docker build -t rlrealtime --no-cache --build-arg BASE=mujoco  .

echo "Built"
