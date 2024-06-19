# import os
# import subprocess

# # Installations :- 
# ######################################################################
# subprocess.run(["pip3", "install", "IPython"])
# subprocess.run(["pip3", "install", "glob"])
# subprocess.run(["pip3", "install", "soundfile"])
# subprocess.run(["pip3", "install", "torchaudio", "--upgrade"])
# ######################################################################

# import sys
# from IPython import display as disp
# import torch
# import torchaudio
# from torchaudio.transforms import Resample
# from denoiser.dsp import convert_audio
# from denoiser import pretrained
# import glob
# import soundfile




# script_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(script_dir)

# def compute_pesq(reference_file, degraded_file):
#     # Load audio files using torchaudio
#     reference_signal, sr_ref = torchaudio.load(reference_file)
#     degraded_signal, sr_deg = torchaudio.load(degraded_file)

#     # Resample signals if sample rates do not match
#     if sr_ref != sr_deg:
#         resample = Resample(sr_deg, sr_ref)
#         degraded_signal = resample(degraded_signal)

#     # Ensure single channel (convert to mono if stereo)
#     reference_signal = reference_signal.mean(dim=0, keepdim=True)
#     degraded_signal = degraded_signal.mean(dim=0, keepdim=True)

#     # Compute PESQ score
#     #pesq_score = pesq.pesq(sr_ref, reference_signal.numpy(), degraded_signal.numpy())

#     return pesq_score

# def dns_64(input_file):
#     output_dir = "."
#     LIST_OF_AUDIO_FILES = glob.glob(input_file)
#     print(LIST_OF_AUDIO_FILES)
#     model = pretrained.dns64()
#     for i in LIST_OF_AUDIO_FILES:
#         print(str(i))
#         file_name = os.path.basename(i)
#         if "Cleaned_denoiser_facebook_" in file_name:
#             continue
#         wav, sr = torchaudio.load(str(i))
#         wav = convert_audio(wav, sr, model.sample_rate, model.chin)  # Use convert_audio directly
#         with torch.no_grad():    
#             denoised = model(wav[None])[0]
#             disp.display(disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate))
#             os.makedirs(output_dir, exist_ok=True)
#             output_file = f"{output_dir}/Cleaned_denoiser_facebook_{file_name}"
#             torchaudio.save(output_file, denoised.data.cpu(), model.sample_rate)

# degraded_file = "./1.wav"
# dns_64(degraded_file)

import os
import subprocess
import sys
from IPython import display as disp
import torch
import torchaudio
from torchaudio.transforms import Resample
from denoiser.dsp import convert_audio
from denoiser import pretrained
import glob
import soundfile
from pystoi import stoi
from enhancer.dsp import convert_audio as convert_audio_enhancer
from enhancer import pretrained as pretrained_enhancer

# Installations ###################################################################
subprocess.run(["pip3", "install", "IPython"])
subprocess.run(["pip3", "install", "glob2"])
subprocess.run(["pip3", "install", "soundfile"])
subprocess.run(["pip3", "install", "torchaudio", "--upgrade"])
subprocess.run(["pip3", "install", "transformers", "--upgrade"])
subprocess.run(["pip3", "install", "pystoi"])
####################################################################################

# Import required libraries
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def dns_64(input_file, output_dir, model_id="addy88/wav2vec2-kannada-stt", target_lang="kn"):
    LIST_OF_AUDIO_FILES = glob.glob(input_file)
    model = pretrained.dns64()
    enhancer_model = pretrained_enhancer.se64_plus()
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    sr_model = Wav2Vec2ForCTC.from_pretrained(model_id)

    for audio_file in LIST_OF_AUDIO_FILES:
        file_name = os.path.basename(audio_file)
        if "Cleaned_denoiser_facebook_" in file_name:
            continue

        wav, sr = torchaudio.load(audio_file)
        wav = convert_audio(wav, sr, model.sample_rate, model.chin)  # Use convert_audio directly

        with torch.no_grad():
            denoised = model(wav[None])[0]
            disp.display(disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate))
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/Cleaned_denoiser_facebook_{file_name}"
            torchaudio.save(output_file, denoised.data.cpu(), model.sample_rate)

            # Perform speech enhancement on the denoised audio
            enhanced = enhancer_model(denoised[None])[0]
            disp.display(disp.Audio(enhanced.data.cpu().numpy(), rate=enhancer_model.sample_rate))
            enhanced_output_file = f"{output_dir}/Enhanced_denoiser_facebook_{file_name}"
            torchaudio.save(enhanced_output_file, enhanced.data.cpu(), enhancer_model.sample_rate)

            # Perform speech recognition on the enhanced audio
            inputs = processor(enhanced.data.cpu().numpy(), return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
            with torch.no_grad():
                logits = sr_model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            print("Transcription:", transcription)

            # Compute STOI score
            enhanced_np = enhanced.data.cpu().numpy()[0]
            wav_np = wav.cpu().numpy()[0]
            stoi_score = stoi(wav_np, enhanced_np, enhancer_model.sample_rate, extended=False)
            print(f"STOI Score: {stoi_score}")

if __name__ == "__main__":
    input_file = r"./INPUT\AUDIO\Rev.mp3"  # Corrected for Windows path
    output_dir = "./OUTPUT/AUDIO"  # Specify the output directory
    model_id = "addy88/wav2vec2-kannada-stt"  # Specify the model ID for speech recognition
    target_lang = "kan"  # Specify the target language code

    dns_64(input_file, output_dir, model_id, target_lang)
