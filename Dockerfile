FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git zip wget libgl1 libglib2.0-0 vim tmux ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Crear entorno con Python 3.10
RUN conda create -n bin-flavr python=3.10 -y

SHELL ["conda", "run", "-n", "bin-flavr", "/bin/bash", "-c"]

COPY requirements.txt /tmp/requirements.txt

# Usar PyTorch con CUDA 11.8
RUN pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

WORKDIR /app
ADD . /app
RUN mkdir -p /app/input /app/output

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate bin-flavr" >> ~/.bashrc

CMD ["bash"]
