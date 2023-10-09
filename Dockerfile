FROM condaforge/miniforge3:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils gcc g++ openssh-server cmake build-essential

RUN apt-get install -y bzip2 wget gnupg dirmngr apt-transport-https \
	ca-certificates openssh-server && \
    apt-get clean

#setup ssh
RUN mkdir /var/run/sshd && \
    echo 'root:root_pwd' |chpasswd && \
    sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' \
    /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config && \
    mkdir /root/.ssh

#remove setup leftovers
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# IOP4 setup for testing
RUN . /opt/conda/etc/profile.d/conda.sh \
    && mamba create -n iop4-test python=3.10 \
    && conda activate iop4-test

# Clone the repository and checkout to the branch of current PR
COPY . /iop4

WORKDIR /iop4

# Install IOP4
RUN pip install .

# Run tests
RUN make test