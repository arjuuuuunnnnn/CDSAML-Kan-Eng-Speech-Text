import subprocess
packages = [
    "torch",
    "indic-nlp-library",
    "sacremoses",
    "regex",
    "pandas",
    "mock",
    "sacrebleu",
    "mosestokenizer",
    "ctranslate2",
    "gradio",
    "nltk",
    "sentencepiece",
    "fairseq",
    "numba",
    "pynvml",
    "koila",
    "tensorboardX",
    "indicnlp",
    "sacremoses"
]

# Iterate through the packages and install them using pip3.10
for package in packages:
    subprocess.run(["pip3.10", "install", package])

import torch
import time
import gc
from inference.engine import Model
ckpt_dir="./indic-en-preprint/fairseq_model/"
model=Model(ckpt_dir,model_type="fairseq")
sents=["ಹೌದು","ನಾಚಿಕೆತರನಾನುಗೊತ್ತು","ನೀವುರಿಪೋರ್ಟರ್"]
src_lang,tgt_lang="kan_Knda","eng_Latn"
print(model.batch_translate(sents,src_lang,tgt_lang))