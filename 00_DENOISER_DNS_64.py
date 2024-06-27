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

# Installations
######################################################################
subprocess.run(["pip3", "install", "IPython"])
subprocess.run(["pip3", "install", "glob2"])
subprocess.run(["pip3", "install", "soundfile"])
subprocess.run(["pip3", "install", "torchaudio", "--upgrade"])
subprocess.run(["pip3", "install", "transformers", "--upgrade"])
subprocess.run(["pip3", "install", "pystoi"])
######################################################################

# Import required libraries
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def dns_64(input_file, output_dir, model_id="addy88/wav2vec2-kannada-stt", target_lang="kn"):
    LIST_OF_AUDIO_FILES = glob.glob(input_file)
    model = pretrained.dns64()
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    sr_model = Wav2Vec2ForCTC.from_pretrained(model_id)

    for audio_file in LIST_OF_AUDIO_FILES:
        file_name = os.path.basename(audio_file)
        if "Cleaned_denoiser_facebook_1" in file_name:
            continue

        wav, sr = torchaudio.load(audio_file)
        wav = convert_audio(wav, sr, model.sample_rate, model.chin)  # Use convert_audio directly

        with torch.no_grad():
            denoised = model(wav[None])[0]
            disp.display(disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate))
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/Cleaned_denoiser_facebook_1{file_name}"
            torchaudio.save(output_file, denoised.data.cpu(), model.sample_rate)

            # Perform speech recognition on the denoised audio
            inputs = processor(denoised.data.cpu().numpy(), return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
            with torch.no_grad():
                logits = sr_model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            print("Transcription:", transcription)

            # Compute STOI score
            denoised_np = denoised.data.cpu().numpy()[0]
            wav_np = wav.cpu().numpy()[0]
            stoi_score = stoi(wav_np, denoised_np, model.sample_rate, extended=False)
            print(f"STOI Score: {stoi_score}")

if __name__ == "__main__":
    input_file = r"./INPUT\AUDIO\audio.mp3"  # Corrected for Windows path
    output_dir = "./OUTPUT/AUDIO"  # Specify the output directory
    model_id = "addy88/wav2vec2-kannada-stt"  # Specify the model ID for speech recognition
    target_lang = "kan"  # Specify the target language code

    dns_64(input_file, output_dir, model_id, target_lang)





