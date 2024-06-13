#######################################################
#https://huggingface.co/facebook/mms-tts-kan
#https://huggingface.co/facebook/mms-tts-eng
#docker build . -t kagapa_tts
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it kagapa_tts --volume .\:/TTS /bin/bash -c  "python3 TTS.py --input_text "ಕನ್ನಡದಲ್ಲಿ ನಮ್ಮ ಭಾಷೆ ಬಹು ಸುಂದರವಾದುದು."  --output kan.wav --language Kannada"
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it kagapa_tts --volume .\:/TTS /bin/bash -c  "python3 TTS.py --input_text "Hi,How are you?"  --output eng.wav --language English"
#######################################################

#OS :-
#######################################################
FROM ubuntu:latest
WORKDIR /TTS
#FROM nvcr.io/nvidia/pytorch:24.03-py3
#Below Error is got from above FROM
# /usr/local/lib/python3.10/dist-packages/transformer_engine_extensions.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
#######################################################

#INSTALLATIONS :-
#######################################################
RUN apt-get update && apt-get install wget -y
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh && echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda create -y -n tts_env python=3.11 && echo "conda activate tts_env" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate tts_env" >> ~/.bashrc
RUN conda install torchaudio -c pytorch -c nvidia -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/ -y
RUN conda install pytorch-cuda -c pytorch -c nvidia -c https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/linux-64/ -y
RUN pip3 install --upgrade transformers 
RUN pip3 install --upgrade accelerate
#######################################################

#EXECUTION :-
#######################################################
#COPY TTS.py TTS.py
CMD ["/bin/bash"]
#######################################################