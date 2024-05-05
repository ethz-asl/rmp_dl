FROM ubuntu:20.04

# Use bash instead of sh (sh doesn't have source)
SHELL ["/bin/bash", "-c"] 
# Dont ask for input
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --yes --no-install-recommends  \
	libosmesa6-dev \
    python3.8   \ 
    virtualenv  \
    python3-pip python3-setuptools  \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev  \
    build-essential  \
    git ssh  \
    wget \
    python3.8-dev \
    libglu1-mesa-dev

RUN ln -s /usr/bin/python3.8 /usr/bin/python

WORKDIR /opt

RUN pip install numpy matplotlib

# We use v16 because v17 has this issue: https://github.com/isl-org/Open3D/issues/6009 (supposedly fixed on master, but dont want to try that out)
RUN git clone https://github.com/isl-org/Open3D.git --branch v0.16.0 --depth 1
WORKDIR /opt/Open3D

RUN  wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.24.1 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

RUN mkdir build 
WORKDIR /opt/Open3D/build
RUN cmake -DENABLE_HEADLESS_RENDERING=ON \
                 -DBUILD_GUI=OFF \
                 -DUSE_SYSTEM_GLEW=OFF \
                 -DUSE_SYSTEM_GLFW=OFF \
                 ..

RUN make -j 4

# Somehow if we don't upgrade pip the make install-pip-package line does not work 
RUN python -m pip install --upgrade pip
RUN which pip
RUN pip --version
RUN make install-pip-package

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /

# Configure Mesa EGL for headless rendering
ENV EGL_PLATFORM=surfaceless
ENTRYPOINT ["python", "/opt/entry/render_world_and_trajectories.py"]

