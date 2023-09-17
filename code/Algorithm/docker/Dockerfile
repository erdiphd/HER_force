
FROM nvidia/cuda:11.4.0-base-ubuntu20.04 

RUN apt-get update && apt-get install --no-install-recommends -y 
    

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  locales \
  git \
  libgl1-mesa-dri \
  menu \
  python \
  pip \
  mesa-utils \
  nano \
  net-tools \
  openbox \
  openssh-server \
  sudo \
  supervisor \
  terminator \
  tint2 \
  vim \
  x11-xserver-utils \
  x11vnc \
  xinit \
  xserver-xorg-video-dummy \
  xserver-xorg-input-void \
  websockify && \
  pip install supervisor-stdout && \
  apt-get -y clean

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    mpich \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN locale-gen en_US.UTF-8 en_GB.UTF-8 de_DE.UTF-8 de_DE@euro
RUN echo 'LANG=en_US.UTF-8' >> /etc/locale.conf
# RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user

# RUN adduser --disabled-password \
# --gecos '' user
# RUN adduser user sudo

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> \
/etc/sudoers


# Create user
RUN useradd -s /bin/bash -d /home/user/ -m -G sudo user


RUN sudo mkdir -p /home/user/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -zxf mujoco.tar.gz --directory /home/user/.mujoco \
    && rm mujoco.tar.gz


# Mini Anaconda Installation

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -u -p /home/user/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

USER user

ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so
ENV MUJOCO_GL "glfw"

# # This lines are only for myself
# COPY ./mjkey.txt /home/user/.mujoco/
# COPY ./mjkey.txt /home/user/.mujoco/mujoco200/bin/

# RUN git clone https://github.com/openai/gym /home/user/hgg_test/gym



RUN /bin/bash -c "source /home/user/conda/bin/activate && \ 
                  conda create --name her python=3.7 -y"
RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  conda activate her && \
                  pip install numpy && pip install cffi lockfile imageio glfw tensorflow==1.14 cython && pip install mujoco_py && pip install beautifultable==0.7.0"

RUN /bin/bash -c "source /home/user/conda/bin/activate && \
                  conda activate her && \
                  pip install numpy-quaternion gym==0.15.4 click joblib mpi4py scipy  protobuf==3.19 scikit-learn pyyaml pyquaternion "
    
ENV DISPLAY :0

VOLUME [ "/home/user/her/" ]


ENV mujoco_env "FetchPickAndPlace-v1"
ENV log_path "log/train1"
ENV n_epochs "200"
ENV num_cpu "1"
ENV prioritization "contact_energy"
ENV reward_type "sparse"

COPY startup.sh /home/user/
# RUN /home/user/.conda/envs/HGG/bin/python /home/user/HGG/train.py
ENV PATH /home/user/.conda/envs/her/bin:$PATH
CMD ["bin/bash", "home/user/startup.sh"]
