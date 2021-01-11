docker run \
    -it \
    --rm \
    --gpus all \
    -v $(pwd):/workspace \
    torch-gpu-py3.8 \
    /bin/bash

