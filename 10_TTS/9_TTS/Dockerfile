#######################################################
#docker build . -t tts
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it tts
#https://docs.coqui.ai/en/latest/docker_images.html

#######################################################
#FROM nvcr.io/nvidia/pytorch:24.01-py3
#FROM python:3.12
FROM ubuntu:23.10
WORKDIR /TTS
RUN apt update && apt upgrade -y
#RUN apt install software-properties-common
#RUN add-apt-repository universe
#RUN apt update && apt upgrade -y
RUN apt-get install -y python3.11 python3-pip
RUN python3 -m pip install --upgrade pip --break
RUN pip3 install --upgrade pip --break
RUN apt install wget -y 
RUN pip3 install scipy --break
RUN pip3 install transformers --break
RUN pip3 install --upgrade transformers accelerate --break
RUN pip3 install torchaudio --break
COPY TTS.py TTS.py
RUN python3.11 TTS.py
CMD ["/bin/bash"]