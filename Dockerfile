#DOCKER BUILD COMMAND :-
###################################################
#docker build . -t kagapa_tools_kit
###################################################

# TEST - "ಕನ್ನಡದಲ್ಲಿ ನಮ್ಮ ಭಾಷೆ ಬಹು ಸುಂದರವಾದುದು."

#TO OPEN CONTAINER :-
###################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it  kagapa_tools_kit
###################################################

#TO SAVE CONTAINER :-
###################################################
#docker commit $(docker ps -q) kagapa_tools_kit
###################################################

#01_DENOISER_DEEP_FILTER RUN COMMAND :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-2 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "deepFilter ./INPUT/AUDIO/KANANDA_TEST_WAV.wav"
################################################################################################

#03_DENOISER_DEMUCS RUN COMMAND :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "demucs ./INPUT/AUDIO/KANANDA_TEST_WAV.wav"
################################################################################################

#04_DENOISER_DNS_64 RUN COMMAND :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "python3 /KAGAPA_TOOLS_KIT/04_DENOISER_DNS_64.py"
################################################################################################

#05 TO DO 

#06_ASR_WHISPHER RUN COMMAND :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "whisper  ./INPUT/AUDIO/KANANDA_TEST_WAV.wav  --model large-v3 --language kn"
################################################################################################

#07_ASR_FB_MMS_1B_PIPELINE RUN COMMAND  :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "python3 /07_ASR_FB_MMS_1B_PIPELINE/run.py_WORKING  "
################################################################################################

# 08_ASR_SEAMLESS_FB  RUN COMMAND:-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it kagapa_tools_kit  /bin/bash -c "m4t_predict ./audio.wav --task s2tt --tgt_lang kan --output_path kan.txt --model_name seamlessM4T_large"
################################################################################################


# 09_ASR_FB_CODE RUN COMMAND :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "m4t_predict --task s2st --model_name seamlessM4T_medium --tgt_lang eng 'path/to/input.wav' --output_path output`date +%Y-%m-%d.%H:%M:%S`.wav"
################################################################################################

#14_MT_ITRANS RUN COMMAND :-
################################################################################################
#pip install torch
#pip install indic-nlp-library 
#pip install sacremoses 
#pip install regex 
#pip install pandas 
#pip install mock 
#pip install sacrebleu
#pip install mosestokenizer 
#pip install ctranslate2 
#pip install gradio 
#pip install nltk 
#pip install sentencepiece 
#pip install fairseq 
#pip install numba 
#pip install pynvml 
#pip install koila
#pip install tensorboardX
#pip install indicnlp
#pip install sacremoses
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "python3.10 /14_MT_ITRANS/FROM_WEBSITE.py"
################################################################################################

#15_MT_WHISPHER RUN COMMAND :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "whisper  ./INPUT/AUDIO/KANANDA_TEST_WAV.wav  --model large-v3 --language kn --task translate"
################################################################################################

#17_NEMO ASR RUN COMMAND :-
################################################################################################
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it nemo_asr /bin/bash -c "python3 /NEMO_ASR/nemo_asr.py"
################################################################################################

#Dockerfile:-
###################################################################################################################################################
###################################################
FROM nvcr.io/nvidia/pytorch:24.03-py3
###################################################

#SETUP THE OS :-
################################################################
RUN apt-get update && apt upgrade -y
RUN python3 -m pip install --upgrade pip
################################################################

#ENVIRONMNET SETUP :-
####################################################
WORKDIR /
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE -1
ENV PYTHONUNBUFFERED 0
ENV PYTHONIOENCODING=utf8
####################################################

#LIST OF DIRECTORIES TO COPY  :-
####################################################
#COPY 04_DENOISER_DNS_64 04_DENOISER_DNS_64
#COPY 07_ASR_FB_MMS_1B_PIPELINE 07_ASR_FB_MMS_1B_PIPELINE
#COPY 09_ASR_FB_CODE 09_ASR_FB_CODE
#COPY 14_MT_ITRANS 14_MT_ITRANS
#COPY 12_ASR_ESPNET_FL 12_ASR_ESPNET_FL
#COPY 17_NEMO_ASR 17_NEMO_ASR
####################################################

#ALL INSTALLATIONS :-
####################################################
RUN apt-get install -y wget 
RUN apt-get install -y bzip2 
RUN apt-get install -y curl
RUN apt install -y rustc
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN source "$HOME/.cargo/env"
RUN pip3 install -U demucs 
RUN pip3 install torchaudio
RUN pip3 install denoiser
RUN pip3 install soundfile
RUN pip3 install ipython
RUN apt install python3-venv -y
RUN pip3 install IPython --break
RUN pip3 install denoiser -U --break

## NEED TO DO THIS :-
# OMEGACONF :-
######################################################
RUN git clone https://github.com/omry/omegaconf.git
RUN pip3 install -r ./omegaconf/requirements/base.txt
RUN pip3 install -r ./omegaconf/requirements/coverage.txt
RUN pip3 install -r ./omegaconf/requirements/dev.txt
RUN pip3 install -r ./omegaconf/requirements/docs.txt
RUN apt install -y default-jdk  --fix-missing
WORKDIR /omegaconf
#RUN python ./setup.py install
#RUN python3 ./setup.py install 
RUN pip3 install . 
WORKDIR /
######################################################

# DEEPFILTERNET INSTALLATION :-
################################################################
RUN pip3 install torchaudio -f https://download.pytorch.org/whl/gpu/torch_stable.html --break-system-packages
RUN pip3 install deepfilternet --break-system-packages
################################################################

# DEMUCS INSTALLATION :-
################################################################
RUN pip3 install -U demucs --break-system-packages
################################################################

# WHISPHER INSTALLATION :-
################################################################
RUN pip3 install -U openai-whisper
RUN pip3 install git+https://github.com/openai/whisper.git 
RUN pip3 install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
RUN apt update && apt install ffmpeg -y
RUN pip3 install setuptools-rust
################################################################

# 07_ASR_FB_MMS_1B_PIPELINE
################################################################
RUN pip3 install torchaudio
RUN pip3 install transformers
################################################################

# 12_ASR_ESPNET_FL INSTALLATIONS :-
####################################################
RUN pip3 install --upgrade accelerate
RUN pip3 install espnet_model_zoo
####################################################



# 08_ASR_SEAMLESS_FB
# NO NEED OF RUNDIR 
####################################################
RUN pip3 install fairseq2 
RUN pip3 install git+https://github.com/facebookresearch/seamless_communication.git --break
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
RUN brew install arrayfire
RUN brew install libsndfile
####################################################




# 09_ASR_FB_CODE :-
################################################################
WORKDIR /09_ASR_FB_CODE
#RUN python3 -m venv asr_fb_code
#RUN source asr_fb_code/bin/activate
# 1126.9 ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE. If you have updated the package versions, please update the hashes. Otherwise, examine the package contents carefully; someone may have tampered with them.
#1126.9     unknown package:
#1126.9         Expected sha256 9d264c5036dde4e64f1de8c50ae753237c12e0b1348738169cd0f8a536c0e1e0
#1126.9              Got        dea4c85e1b491da4195a485f91381eae6bba362a0396304fa723f73b2d3053d5
#RUN  pip3 install  torchvision --break
#RUN pip3 install fairseq2n --break
#RUN  pip3 install  fairseq2==0.2.0 --break 
RUN  pip3 install pydub --break 
RUN  pip3 install yt-dlp --break 
#RUN  git clone https://github.com/facebookresearch/seamless_communication.git
#RUN  cd seamless_communication
#RUN pip3 install -r ./requirements.txt --break
#RUN python3 setup.py install 
#RUN  pip3 install  . --break 
RUN  pip3 install  protobuf --break
RUN  apt-get install libsox-dev -y
RUN  apt-get install libsox-fmt-all -y
RUN  pip3 install  deepspeed  --break
RUN  pip3 install  scipy  --break
RUN  pip3 install  chardet  --break
RUN  pip3 install  torchaudio  --break
RUN  pip3 install  transformers  --break
RUN  pip3 install  sentencepiece  --break
#RUN source deactivate
#WORKDIR /09_ASR_FB_CODE
#RUN deactivate
#RUN source asr_fb_code/bin/deactivate
RUN cd .. 
################################################################



# 14_MT_TRANS INSTALLATIONS :-
####################################################
WORKDIR /14_MT_ITRANS
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.9 -y
RUN apt install python3.9-venv -y
#RUN apt-get update && apt-get install -y python3.9 python3.9-venv
RUN python3.9 -m venv 14_MT_ITANS2
ENV PATH="/14_MT_ITANS2/bin:$PATH"
ENV root_dir=/14_MT_ITRANS
#ENV PATH /opt/conda/bin:$PATH
ENV INDIC_RESOURCES_PATH=$root_dir/indic_nlp_resources
#RUN apt-get update && \
#    apt-get install -y git wget bzip2 && \
#    rm -rf /var/lib/apt/lists/*
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    bash ~/miniconda.sh -b -p /opt/conda && \
#    rm ~/miniconda.sh
#RUN /opt/conda/bin/conda create -y -n itv2 python=3.9 && \
#    echo "conda activate itv2" > ~/.bashrc
#SHELL ["/bin/bash", "-c", "source activate itv2"]
#RUN source ~/.bashrc
#RUN conda activate itv2
RUN pip3 install indic-nlp-library  --break
RUN pip3 install sacremoses   --break
RUN pip3 install regex   --break
RUN pip3 install pandas   --break
RUN pip3 install mock   --break
RUN pip3 install transformers   --break
RUN pip3 install sacrebleu  --break
RUN pip3 install mosestokenizer   --break
RUN pip3 install ctranslate2   --break
RUN pip3 install nltk   --break
RUN pip3 install sentencepiece   --break 
RUN pip3 install fairseq   --break
RUN pip3 install numba   --break  
RUN pip3 install pynvml   --break
RUN pip3 install koila  --break
RUN pip3 install tensorboardX  --break
RUN python3 -c "import nltk; nltk.download('punkt')"
RUN pip3 install indicnlp  --break
RUN pip3 install sacremoses  --break
####################################################

# 17_NEMO_ASR INSTALLATION :-
###################################################
RUN apt install python3.11 -y
RUN apt install python3.11-venv -y
#RUN apt-get update && apt-get install -y python3.9 python3.9-venv
RUN python3.11 -m venv 17_NEMO_ASR
ENV PATH="/17_NEMO_ASR/bin:$PATH"
RUN pip3 install  huggingface_hub>=0.20.3 --break
RUN pip3 install  numba --break
RUN pip3 install  numpy>=1.22 --break
RUN pip3 install  python-dateutil --break
RUN pip3 install  ruamel.yaml --break
RUN pip3 install  scikit-learn --break
RUN pip3 install  setuptools>=65.5.1 --break
RUN pip3 install  tensorboard --break
RUN pip3 install  text-unidecode --break
RUN pip3 install  tqdm>=4.41.0 --break
RUN pip3 install  triton --break
RUN pip3 install  wrapt --break
RUN pip3 install  braceexpand --break
RUN pip3 install  editdistance --break
RUN pip3 install  g2p_en --break
RUN pip3 install  ipywidgets --break
RUN pip3 install  jiwer --break 
RUN pip3 install  kaldi-python-io --break
RUN pip3 install  kaldiio --break
RUN pip3 install  lhotse>=1.20.0 --break
RUN pip3 install  librosa>=0.10.0 --break
RUN pip3 install  marshmallow --break
RUN pip3 install  matplotlib --break
RUN pip3 install  packaging --break
RUN pip3 install  pyannote.core --break
RUN pip3 install  pyannote.metrics --break
RUN pip3 install  pydub --break
RUN pip3 install  pyloudnorm --break
RUN pip3 install  resampy --break
RUN pip3 install  ruamel.yaml --break
RUN pip3 install  scipy>=0.14 --break
RUN pip3 install  soundfile --break
RUN pip3 install  sox --break
RUN pip3 install  datasets --break
RUN pip3 install  inflect --break
RUN pip3 install  pandas --break
RUN pip3 install  sacremoses>=0.0.43 --break
RUN pip3 install  sentencepiece --break
RUN pip3 install unidecode --break
RUN apt --fix-broken install -y
RUN apt update -y && apt install libsndfile1 -y 
RUN apt-get install  ffmpeg -y 
RUN pip3 install text-unidecode  --break
RUN pip3 install matplotlib>=3.3.2  --break
RUN pip3 install librosa --break
RUN pip3 install pytorch_lightning --break
RUN pip3 install transformers --break
RUN pip3 install webdataset --break
RUN pip3 install hydra-core --break
RUN pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/  --break
# WORKS in CONTAINDER
#RUN export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
#RUN pip3 install onnx>=1.7.0 --break
#RUN pip3 install git+https://github.com/NVIDIA/NeMo.git --break
###################################################
#### END OF ALL INSTALLATIONS #################################


######################################################
#WORKDIR /02_DENOISER_DeepXi-master__DOCKER_PRESENT
######################################################

# 04 DENOISER DNS 64 :-
######################################################
#WORKDIR /04_DENOISER_DNS_64
#RUN python3 dns64.py
######################################################


#WORKDIR /05_DENOISER_espnet__NEED_TO_CONVERT_JUPYTER_TO_Dockerfile
######################################################
######################################################



# ASR FB_MMS_1B :-
######################################################
################################################################################################
#docker build . -t fb_mms_1b_1
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT -it kagapa_tools_kit  /bin/bash -c "python run.py_WORKING"
# HOW TO INPUT THE MODEL FROM docker run commnad ?
################################################################################################
################################################################################################
#ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONUNBUFFERED 0  
#RUN apt update
#RUN apt upgrade -y
#RUN pip3 install torchaudio
#RUN pip3 install transformers

#WORKDIR /07_ASR_FB_MMS_1B_PIPELINE
#RUN python3 -m venv fb_mms_1b
#SHELL ["/bin/bash", "-c"]
#RUN source fb_mms_1b/bin/activate
#RUN python3 ./run.py_WORKING
#RUN source deactivate
#RUN cd ..
###################################################


# 08_ASR_SEAMLESS_FB :-
######################################################
################################################################################################
#https://orth.uk/running-seamlessm4t-ai-locally/
#https://brandolosaria.medium.com/setting-up-meta-ais-seamlessm4t-massively-multilingual-multimodal-machine-translation-model-5d2904956761
#docker build . -t seamless_m4t
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it kagapa_tools_kit  /bin/bash -c "m4t_predict ./audio.wav --task s2tt --tgt_lang kan --output_path kan.txt --model_name seamlessM4T_large"
################################################################################################
#ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONUNBUFFERED 0
#RUN apt update
#RUN apt upgrade -y
#RUN apt install git -y
#RUN apt install python3-venv -y
#RUN python3 -m venv seamless_m4t
#RUN source seamless_m4t/bin/activate
#RUN git clone https://github.com/facebookresearch/seamless_communication.git
#WORKDIR /SEAMLESS_FB/seamless_communication
#RUN pip3 install torch 
#RUN pip3 install fairseq2 
#RUN pip3 install  .
#RUN pip3 install -e . 
#RUN python3 setup.py install 
#RUN python setup.py install
#RUN cd ..
#RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
#RUN brew install arrayfire
#RUN brew install libsndfile
#COPY . . 
#CMD ["/bin/bash"]
#WORKDIR /08_ASR_SEAMLESS_FB
######################################################


# ASR FB CODE :-
################################################################################################
#https://orth.uk/running-seamlessm4t-ai-locally/
#https://brandolosaria.medium.com/setting-up-meta-ais-seamlessm4t-massively-multilingual-multimodal-machine-translation-model-5d2904956761
#docker build . -t seamless_m4t
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it  kagapa_tools_kit /bin/bash -c "m4t_predict ./audio.wav --task s2tt --tgt_lang kan --output_path kan.txt --model_name seamlessM4T_large"
################################################################################################
#WORKDIR /09_ASR_FB_CODE
#ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONUNBUFFERED 0
#WORKDIR /SEAMLESS_FB
#RUN apt update
#RUN apt upgrade -y
#RUN apt install git -y
#RUN apt install python3-venv -y
#RUN python3 -m venv seamless_m4t
#RUN source seamless_m4t/bin/activate
#RUN git clone https://github.com/facebookresearch/seamless_communication.git
#RUN cd ./seamless_communication
#WORKDIR /SEAMLESS_FB/seamless_communication
#RUN pip3 install torch 
#RUN pip3 install fairseq2 
#RUN pip3 install  .
#RUN pip3 install -e . 
#RUN python3 setup.py install 
#RUN python setup.py install
#RUN cd ..
#RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
#RUN brew install arrayfire
#RUN brew install libsndfile
#COPY . . 
#CMD ["/bin/bash"]
######################################################

# 09_ASR_FB_CODE :-
######################################################
################################################################################################
#docker build . -t fb_code
#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it  kagapa_tools_kit
################################################################################################
#ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONUNBUFFERED 0
#RUN apt-get update
#RUN apt upgrade -y
#WORKDIR /app
#RUN  pip3 install torch 
#RUN  pip3 install  torchvision
#RUN  pip3 install  fairseq2 
#RUN  pip3 install pydub 
#RUN  pip3 install yt-dlp 
#RUN  git clone https://github.com/facebookresearch/seamless_communication.git
#RUN  cd seamless_communication
#RUN  pip3 install  . 
#RUN  pip3 install  protobuf
#RUN  cd ..
#RUN  apt-get install libsox-dev libsox-fmt-all
#RUN  pip3 install  deepspeed
#RUN  pip install   deepspeed
#RUN  pip3 install  scipy
#RUN  pip3 install  chardet
#RUN  pip3 install  torchaudio
#RUN  pip3 install  transformers
#RUN  pip3 install  sentencepiece
#RUN  python3 SCRIPT_WORKING.py_GOOD
#COPY . . 
#CMD ["/bin/bash"]
#WORKDIR /09_ASR_FB_CODE
#RUN  python3 SCRIPT_WORKING.py_GOOD
#RUN cd ..
###################################################

# ASR ESPNET
######################################################
#RUN pip3 install --upgrade accelerate
#RUN pip3 install torch
#RUN pip3 install espnet_model_zoo
#WORKDIR /12_ASR_ESPNET_FL
#RUN python3 asr.py
#RUN cd ..
######################################################

# MT ITRNS2
######################################################
##################################################################################################################
# docker build . -t itrans2
# docker run --gpus all  -it itrans
# conda init
# source ~/.bashrc
# conda activate itv2
# bash SCRIPT_ALL.csh
####################### OTHER COMMANDS :-
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\indic-en-preprint:/IndicTrans2 -it  kagapa_tools_kit /bin/bash -c "bash /IndicTrans2/SCRIPT_ALL.csh"
##################################################################################################################
#WORKDIR /14_MT_ITRANS
#ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONIOENCODING=utf8
#ENV root_dir=/IndicTrans2
#ENV PATH /opt/conda/bin:$PATH
#ENV INDIC_RESOURCES_PATH=$root_dir/indic_nlp_resources
#RUN apt-get update && \
#    apt-get install -y git wget bzip2 && \
#    rm -rf /var/lib/apt/lists/*
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#    bash ~/miniconda.sh -b -p /opt/conda && \
#    rm ~/miniconda.sh
#RUN /opt/conda/bin/conda create -y -n itv2 python=3.9 && \
#    echo "conda activate itv2" > ~/.bashrc
#SHELL ["/bin/bash", "-c", "source activate itv2"]
#RUN source ~/.bashrc
#RUN conda activate itv2
#RUN git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git /IndicTrans2/indic_nlp_resources
#WORKDIR /IndicTrans2
#RUN pip3 install indic-nlp-library 
#RUN pip3 install sacremoses 
#RUN pip3 install regex 
#RUN pip3 install pandas 
#RUN pip3 install mock 
#RUN pip3 install transformers 
#RUN pip3 install sacrebleu
#RUN pip3 install mosestokenizer 
#RUN pip3 install ctranslate2 
#RUN pip3 install gradio 
#RUN pip3 install nltk 
#RUN pip3 install sentencepiece 
#RUN pip3 install fairseq 
#RUN pip3 install numba 
#RUN pip3 install pynvml 
#RUN pip3 install koila
#RUN pip3 install tensorboardX
#RUN python -c "import nltk; nltk.download('punkt')"
#RUN pip3 install torch
#RUN pip3 install indicnlp
#RUN pip3 install sacremoses
#COPY inference inference
#COPY FROM_WEBSITE.py FROM_WEBSITE.py
#COPY SCRIPT_ALL.csh SCRIPT_ALL.csh 
#COPY indic-en-preprint   indic-en-preprint
#CMD ["/bin/bash"]
######################################################

###################################################
WORKDIR /
#WRITE CODE TO DOWNLOAD THE INDICTANS2 MODEL 
#COPY AUDIOS AUDIOS 
CMD ["/bin/bash"]
###################################################
