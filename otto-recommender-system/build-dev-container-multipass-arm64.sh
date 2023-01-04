#!/bin/bash

export DOCKER_BUILDKIT=1

docker build -t kaggle -f ./otto-recommender-system/dev.arm64.Dockerfile .

if [[ "$OSTYPE" == "darwin"* ]]; then
    xhost +$(multipass list | grep docker-vm | awk '{print $3}')
    docker run -it --rm \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'):0.0 \
        --name otto-recommender-system \
        kaggle \
        /bin/bash
    xhost -$(multipass list | grep docker-vm | awk '{print $3}')
else
    xhost +local:
    docker run -it --rm \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        --name otto-recommender-system \
        kaggle \
        /bin/bash
    xhost -local:
fi
