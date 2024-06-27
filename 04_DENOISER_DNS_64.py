import os
import subprocess

# Installations :- 
######################################################################
subprocess.run(["pip3", "install", "IPython"])
subprocess.run(["pip3", "install", "glob"])
subprocess.run(["pip3", "install", "soundfile"])
subprocess.run(["pip3", "install", "torchaudio", "--upgrade"])
######################################################################

import sys
from IPython import display as disp
import torch
import torchaudio
from torchaudio.transforms import Resample
from denoiser.dsp import convert_audio
from denoiser import pretrained
import glob
import soundfile




script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def compute_pesq(reference_file, degraded_file):
    # Load audio files using torchaudio
    reference_signal, sr_ref = torchaudio.load(reference_file)
    degraded_signal, sr_deg = torchaudio.load(degraded_file)

    # Resample signals if sample rates do not match
    if sr_ref != sr_deg:
        resample = Resample(sr_deg, sr_ref)
        degraded_signal = resample(degraded_signal)

    # Ensure single channel (convert to mono if stereo)
    reference_signal = reference_signal.mean(dim=0, keepdim=True)
    degraded_signal = degraded_signal.mean(dim=0, keepdim=True)

    # Compute PESQ score
    #pesq_score = pesq.pesq(sr_ref, reference_signal.numpy(), degraded_signal.numpy())

    return pesq_score

def dns_64(input_file):
    output_dir = "."
    LIST_OF_AUDIO_FILES = glob.glob(input_file)
    print(LIST_OF_AUDIO_FILES)
    model = pretrained.dns64()
    for i in LIST_OF_AUDIO_FILES:
        print(str(i))
        file_name = os.path.basename(i)
        if "Cleaned_denoiser_facebook_" in file_name:
            continue
        wav, sr = torchaudio.load(str(i))
        wav = convert_audio(wav, sr, model.sample_rate, model.chin)  # Use convert_audio directly
        with torch.no_grad():    
            denoised = model(wav[None])[0]
            disp.display(disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate))
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/Cleaned_denoiser_facebook_{file_name}"
            torchaudio.save(output_file, denoised.data.cpu(), model.sample_rate)

degraded_file = "./1.wav"
dns_64(degraded_file)
