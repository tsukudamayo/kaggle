FROM rust:slim-bullseye

ENV HOME /home
ENV PATH $PATH:$HOME/.cargo/bin
ENV PATH $PATH:$HOME/.local/bin
ENV PYTHON_VERSION 3.9.14

WORKDIR /home

RUN apt update \
    && apt-get install -y software-properties-common \
    wget \
    curl \
    gnupg2 \
    git \
    libssl-dev \
    pkg-config \
    build-essential \
    gnutls-bin \
    && rustup update \
    && rustup component add rustfmt clippy rls rust-analysis rust-src  \
    && mkdir -p ~/.cargo/bin \
    && curl -L https://github.com/rust-lang/rust-analyzer/releases/download/2022-07-04/rust-analyzer-aarch64-unknown-linux-gnu.gz | gunzip -c - > ~/.cargo/bin/rust-analyzer \
    && chmod +x ~/.cargo/bin/rust-analyzer \ 
    && cargo install cargo-edit \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xf Python-${PYTHON_VERSION}.tgz \
    && rm -rf /var/lib/apt/lists/*

# build python
WORKDIR /home/Python-${PYTHON_VERSION}
RUN apt update && apt install -y zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    # for scipy install
    gfortran libopenblas-dev liblapack-dev \
    && ./configure --enable-optimizations \
    && make -j 8 \
    && make install \
    && curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt install -y nodejs \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && pip3 install python-lsp-server pyright \
    && echo 'alias python="python3"' >> ~/.bashrc \
    && echo 'alias pip="pip3"' >> ~/.bashrc \
    && rm -rf /home/Python-${PYTHON_VERSION} \
    && rm /home/Python-${PYTHON_VERSION}.tgz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

CMD ["/bin/bash"]
