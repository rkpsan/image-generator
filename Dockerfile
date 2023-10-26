# Define base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base

# Set shell options and environment variables
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Africa/Johannesburg \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

WORKDIR /

# Install necessary packages and tools
RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    python3.10-venv \
    python3-pip \
    python3-tk \
    python3-dev \
    nodejs \
    npm \
    bash \
    dos2unix \
    git \
    git-lfs \
    ncdu \
    nginx \
    net-tools \
    inetutils-ping \
    openssh-server \
    libglib2.0-0 \
    libsm6 \
    libgl1 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    wget \
    curl \
    psmisc \
    rsync \
    vim \
    zip \
    unzip \
    p7zip-full \
    htop \
    pkg-config \
    plocate \
    libcairo2-dev \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    apt-transport-https \
    ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
# Install Torch, xformers and tensorrt
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir xformers==0.0.21 tensorrt

# Stage 2: Install applications
FROM base as setup




# Set working directory and copy app contents
WORKDIR /app
COPY ./app /app

# Install Python libraries
RUN pip install fastapi uvicorn jupyterlab transformers accelerate && \
  pip install -r requirements.txt

# Add and set permissions for start script
ADD start.sh /
RUN chmod +x /start.sh

# Define default command
CMD [ "/start.sh" ]
