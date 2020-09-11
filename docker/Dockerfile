FROM debian:bullseye
MAINTAINER Tim Molteno "tim@elec.ac.nz"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3-numpy \
    python3-astropy \
    python3-requests \
    python3-tz \
    python3-casacore \
    python3-pip \
    python3-distributed \
    libopenblas-dev

RUN pip3 install tart2ms
RUN pip3 install disko

ENV LANG en_US.UTF-8 
