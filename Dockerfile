FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# replace CUDA version to your CUDA version.
# You can check your CUDA version with below.
# nvcc -Vdocker pull nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get -y install python3-pip git
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# replace CUDA version to your CUDA version.

RUN mkdir workspace
WORKDIR /workspace

RUN pip3 install fastapi uvicorn[standard] fsspec[http]==2023.5.0
COPY nougat_edited /workspace/nougat
WORKDIR /workspace/nougat
RUN pip install pypdfium2==4.30.1
# RUN pip3 install fsspec==2023.5.0
RUN python3 setup.py install
RUN pip3 install transformers==4.38.2
RUN pip3 install albumentations==1.0.0
RUN pip3 install python-multipart
EXPOSE 8503

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8503"]
# Run this using 'docker run -it -d -p <YOUR PORT>:8503 --gpus all <IMAGE NAME>
