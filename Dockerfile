FROM condaforge/miniforge3:latest as base

ENV DEBIAN_FRONTEND=noninteractive

# Basic packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential && apt-get clean

# Remove setup leftovers
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# IOP4 setup for testing
RUN . /opt/conda/etc/profile.d/conda.sh \
    && mamba create -n iop4-test python=3.10 \
    && conda activate iop4-test

# Define $HOME as required by IOP4 tests
ENV HOME /home/testuser

# Location of astrometry files
VOLUME $HOME/iop4

# Copy IOP4 test dataset
COPY iop4testdata.tar.gz $HOME/iop4testdata.tar.gz

# Copy current branch
COPY . /app
WORKDIR /app

# Install IOP4
RUN pip install .[test]

# Run tests
ENTRYPOINT ["pytest"]
