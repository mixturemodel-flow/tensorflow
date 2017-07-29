FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Sebastian Weiss <sebastian13.weiss@tum.de>

# Install tensorflow prerequesites
# Taken and adapted from tensorflow/tensorflow/tools/ci_build/Dockerfile.gpu
COPY tensorflow/tensorflow/tools/ci_build/install/*.sh /install/
RUN /install/install_bootstrap_deb_packages.sh
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    add-apt-repository -y ppa:george-edison55/cmake-3.x
RUN /install/install_deb_packages.sh
RUN /install/install_pip_packages.sh
RUN /install/install_bazel.sh
RUN /install/install_golang.sh

# Set properties, so that ./config does not ask for settings
ENV TF_NEED_MKL 0
ENV CC_OPT_FLAGS -march=native
ENV TF_NEED_JEMALLOC 1
ENV TF_NEED_GCP 0
ENV TF_NEED_HDFS 0
ENV TF_ENABLE_XLA 0
ENV TF_NEED_VERBS 0
ENV TF_NEED_OPENCL 0
ENV TF_NEED_CUDA 1
ENV TF_CUDA_CLANG 0
ENV TF_CUDA_VERSION ""
ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV GCC_HOST_COMPILER_PATH /usr/bin/gcc-4.9
ENV TF_CUDNN_VERSION ""
ENV CUDNN_INSTALL_PATH /usr/local/cuda
ENV TF_CUDA_COMPUTE_CAPABILITIES 5.2
RUN ./configure

# Compile it
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 

