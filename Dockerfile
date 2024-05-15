# Base image
FROM tensorflow/tensorflow:2.15.0-gpu
#python:3.9.18

# Copy files
WORKDIR /workspace
COPY requirements.txt ./

# Environment
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
