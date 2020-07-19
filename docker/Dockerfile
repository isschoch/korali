FROM debian:latest

# update aptitude
RUN apt-get update && apt-get -y upgrade

# install aptitude essentials
RUN apt-get -y install \
    build-essential \
    cmake \
    git \
    vim \
    curl \
    wget \
    pkg-config \
    openmpi-bin \
    openmpi-common \
    libhdf5-openmpi-dev \
    libcairo2-dev \
    libcairo-gobject2 \
    python3-dev \
    python3-pip \
    python3-numpy

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pycairo
RUN python3 -m pip install mpi4py

# add non-root user
RUN useradd -m korali
WORKDIR /home/korali
USER korali

# install korali
RUN git clone https://github.com/cselab/korali.git --recursive
RUN (cd korali && MPICXX=mpic++ ./install --prereqs --jobs=4)
RUN (cd korali/tests && ./run_all_tests.sh)
RUN rm -rf korali/external/_llvm
