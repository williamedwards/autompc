FROM pytorch/pytorch

COPY . /src
WORKDIR /src

# Install general dependencies
RUN pip install -r requirements.txt
RUN pip install coverage

# Install ipopt
RUN conda install ipopt cyipopt -c conda-forge 

# Install mujoco
RUN apt-get update \
    && apt-get -y install curl wget build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev \
    && mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

RUN pip install mujoco_py