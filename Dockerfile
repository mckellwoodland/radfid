# Base image
FROM tensorflow/tensorflow:2.15.0-gpu
#python:3.9.18

# Copy files
WORKDIR /workspace
COPY requirements.txt ./

# Install dependencies
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
