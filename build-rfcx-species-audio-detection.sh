#!/bin/bash

xhost +local:
docker run \
    -it --rm \
    --gpus all \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /media/tsukudamayo/0CE698DCE698C6FC/tmp/data/dataset/rfcx-species-audio-detection/input:/workspace/input \
    -e DISPLAY=$DISPLAY \
    torch-gpu-py3.8 \
    /bin/bash
xhost -local:
