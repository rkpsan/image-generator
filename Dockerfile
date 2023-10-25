# Define base image
ARG BASE_IMAGE=tensorflow/tensorflow:latest-gpu
FROM ${BASE_IMAGE} as dev-base

# Set shell options and environment variables
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
  SHELL=/bin/bash

# Install necessary packages and tools
RUN apt-key del 7fa2af80 && \
  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
  apt-get update --yes && \
  # apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
  # the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
  apt-get upgrade --yes && \
  apt-get install --yes --no-install-recommends \
  wget \
  bash \
  openssh-server && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
  /usr/bin/python3 -m pip install --upgrade pip && \
  pip install jupyterlab ipywidgets

# Set working directory and copy app contents
WORKDIR /app
COPY ./app /app

# Install Python libraries
RUN pip install fastapi uvicorn && \
  pip install -r requirements.txt

# Add and set permissions for start script
ADD start.sh /
RUN chmod +x /start.sh

# Define default command
CMD [ "/start.sh" ]
