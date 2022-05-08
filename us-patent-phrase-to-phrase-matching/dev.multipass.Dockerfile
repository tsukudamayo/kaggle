FROM python:3.9-slim-bullseye

ENV PATH $PATH:/root/.poetry/bin

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python \
    && pip install python-lsp-server \
    && echo 'export DISPLAY="$(ifconfig en0 | grep inet | grep -v inet6 | awk '{print $2}')"' \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
