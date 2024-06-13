## Ongoing...

#### Docker build
```bash
docker build . -t kagapa_tools_kit
```

#### To Open Container
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it  kagapa_tools_kit
```

#### To Save Container
```bash
docker commit $(docker ps -q) kagapa_tools_kit
```

### 01_DENOISER_DEEP_FILTER RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-2 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "deepFilter ./INPUT/AUDIO/KANANDA_TEST_WAV.wav"
```

### 03_DENOISER_DEMUCS RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "demucs ./INPUT/AUDIO/KANANDA_TEST_WAV.wav"
```

### 04_DENOISER_DNS_64 RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "python3 /KAGAPA_TOOLS_KIT/04_DENOISER_DNS_64.py"
```

## TO DO:

#### 06_ASR_WHISPHER RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "whisper  ./INPUT/AUDIO/KANANDA_TEST_WAV.wav  --model large-v3 --language kn"
```

#### 07_ASR_FB_MMS_1B_PIPELINE RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "python3 /07_ASR_FB_MMS_1B_PIPELINE/run.py_WORKING  "
```

#### 08_ASR_SEAMLESS_FB  RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it kagapa_tools_kit  /bin/bash -c "m4t_predict ./audio.wav --task s2tt --tgt_lang kan --output_path kan.txt --model_name seamlessM4T_large"
```

#### 09_ASR_FB_CODE RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "m4t_predict --task s2st --model_name seamlessM4T_medium --tgt_lang eng 'path/to/input.wav' --output_path output`date +%Y-%m-%d.%H:%M:%S`.wav"
```

#### 14_MT_ITRANS RUN COMMAND
```bash
pip install torch
pip install indic-nlp-library 
pip install sacremoses 
pip install regex 
pip install pandas 
pip install mock 
pip install sacrebleu
pip install mosestokenizer 
pip install ctranslate2 
pip install gradio 
pip install nltk 
pip install sentencepiece 
pip install fairseq 
pip install numba 
pip install pynvml 
pip install koila
pip install tensorboardX
pip install indicnlp
pip install sacremoses
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "python3.10 /14_MT_ITRANS/FROM_WEBSITE.py"
```

#### 15_MT_WHISPHER RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v .\:/INPUT kagapa_tools_kit   /bin/bash -c "whisper  ./INPUT/AUDIO/KANANDA_TEST_WAV.wav  --model large-v3 --language kn --task translate"
```

#### 17_NEMO ASR RUN COMMAND
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it nemo_asr /bin/bash -c "python3 /NEMO_ASR/nemo_asr.py"
```


