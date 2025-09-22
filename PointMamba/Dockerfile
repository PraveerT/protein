FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install wheel

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN pip install ninja

ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN cd extensions/chamfer_dist && python setup.py install

RUN cd extensions/emd && python setup.py install

RUN git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch.git
RUN MAX_JOBS=1 cd Pointnet2_PyTorch/pointnet2_ops_lib && python setup.py install

RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

RUN pip install --no-build-isolation causal-conv1d==1.2.1
RUN cd mamba && pip install .

RUN pip install jupyterlab
RUN pip install transformers==4.41.0

WORKDIR /app

CMD ["/bin/bash"]