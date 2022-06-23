FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

MAINTAINER Casimiro de Almeida Barreto (cdab63)

# Set variables
ENV PATH /usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64:/darknet
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

LABEL com.nvidia.volumes.needed="nvidia_driver"

# Get gpg key for repo (bug in image)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install dependencies
RUN apt update
RUN apt -y full-upgrade
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC-3 apt-get -y install tzdata
RUN apt install -y libopencv-dev
RUN apt install -y libomp-dev
RUN apt install -y git
RUN apt install -y nfs-common
RUN apt install -y sshpass

# Clone darknet/yolov4
RUN git clone https://github.com/AlexeyAB/darknet

# Fix darknet Makefile enabling CUDA CUDNN OPENCV OPENMP LIBSO
RUN mv darknet/Makefile darknet/Makefile.bak
RUN cat darknet/Makefile.bak | sed 's/GPU=0/GPU=1/' | sed 's/CUDNN=0/CUDNN=1/' | sed 's/CUDNN_HALF=0/CUDNN_HALF=1/' | sed 's/OPENCV=0/OPENCV=1/' | sed 's/AVX=0/AVX=1/' | sed 's/OPENMP=0/OPENMP=1/' | sed 's/LIBSO=0/LIBSO=1/' > darknet/Makefile

# Build darknet
RUN cd darknet; make

# Custom modules
RUN apt install -y python3-pip python3-opencv
RUN pip install -U --user keras tensorflow-gpu transformers torch datasets