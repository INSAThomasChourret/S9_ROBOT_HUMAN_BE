# Use Debian 12 as base
FROM debian:12

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
# Added 'tmux' and 'python3-pip'
RUN apt-get update -qq && \
    apt-get install -qqy \
    curl \
    gnupg \
    lsb-release \
    git \
    sudo \
    tmux \
    python3-tk python3-pyqt5 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Add Robotpkg key and repository
RUN curl -sSL http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - && \
    echo "deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -cs) robotpkg" > /etc/apt/sources.list.d/robotpkg.list && \
    echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" >> /etc/apt/sources.list.d/robotpkg.list

# Update and install Robotpkg packages
RUN apt-get update -qq && \
    apt-get install -qqy robotpkg-py3*-pinocchio robotpkg-py3*-example-robot-data robotpkg-py3*-qt5-gepetto-viewer-corba && \
    rm -rf /var/lib/apt/lists/*

# Install specific python packages
# Using --break-system-packages is necessary on Debian 12 because we are 
# installing globally without a virtual environment.
RUN pip3 install --no-cache-dir --break-system-packages "numpy<1.27" matplotlib tqdm pyqtgraph


#export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.11/site-packages
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openrobots/lib

RUN echo 'export PATH="$PATH:/opt/openrobots/bin"' >> /root/.bashrc
RUN echo 'export PYTHONPATH="$PYTHONPATH:/opt/openrobots/lib/python3.11/site-packages"' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/openrobots/lib"' >> /root/.bashrc
ENV PATH="$PATH:/opt/openrobots/bin"
ENV PYTHONPATH="$PYTHONPATH:/opt/openrobots/lib/python3.11/site-packages"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/openrobots/lib"

# Set working directory
WORKDIR /root

# Default command
CMD ["/bin/bash"]
