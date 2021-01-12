#!/bin/bash

xhost +local:
docker run \
    -it --rm \
    --gpus all \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    torch-gpu-py3.8 \
    /bin/bash
xhost -local:
