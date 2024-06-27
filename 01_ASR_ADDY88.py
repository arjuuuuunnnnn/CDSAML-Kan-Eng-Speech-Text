# import torchaudio
# import torch
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor

# model_id = "addy88/wav2vec2-kannada-stt"
# target_lang = "kn"

# # Replace with your actual file path
# audio_file = "./INPUT\AUDIO\Rev.mp3"  # Make sure you have a file named "audio.wav" in the same directory

# # Load audio using torchaudio (recommended):
# try:
#     audio, orig_freq = torchaudio.load(audio_file)

#     # Ensure mono channel:
#     audio = audio.mean(dim=0, keepdim=True)

#     # Resample to model-compatible frequency (check documentation):
#     resampler = torchaudio.transforms.Resample(orig_freq, 16_000)  # assuming 16 kHz
#     audio = resampler(audio)

#     # Prepare audio input for the model:
#     audio_input = {"input_values": audio}

# except OSError as e:
#     print(f"Error loading audio file: {e}")
#     exit(1)

# # Load the processor for inference:
# processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)

# # Perform inference using pipeline:
# try:
#     inputs = processor(
#         audio_input["input_values"].numpy(),
#         return_tensors="pt",
#         sampling_rate=16_000
#     )

#     # Load the model for inference:
#     model = Wav2Vec2ForCTC.from_pretrained(model_id)

#     with torch.no_grad():
#         logits = model(inputs.input_values).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)[0]
#     print("Transcription:", transcription)

# except Exception as e:
#     print(f"Error during inference: {e}")
















#--------------------------------------------------------------------------------------------------------------------

import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import jiwer

model_id = "addy88/wav2vec2-kannada-stt"

# Replace with your actual file path
audio_file = "./INPUT/AUDIO/Cleaned_denoiser_facebook_1audio.mp3"  # Ensure correct path format
original_transcript_file = "Kannada_Input_Transcript3.txt"  # Path to the original transcript

# Load audio using torchaudio (recommended):
try:
    audio, orig_freq = torchaudio.load(audio_file)

    # Ensure mono channel:
    audio = audio.mean(dim=0, keepdim=True)

    # Resample to model-compatible frequency (check documentation):
    resampler = torchaudio.transforms.Resample(orig_freq, 16_000)  # assuming 16 kHz
    audio = resampler(audio)

    # Prepare audio input for the model:
    audio_input = {"input_values": audio}

except OSError as e:
    print(f"Error loading audio file: {e}")
    exit(1)

# Load the processor for inference:
processor = Wav2Vec2Processor.from_pretrained(model_id)

# Perform inference using pipeline:
try:
    inputs = processor(
        audio_input["input_values"].numpy(),
        return_tensors="pt",
        sampling_rate=16_000
    )

    # Load the model for inference:
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print("Original Transcription:", transcription)

    # Remove <s> tokens from the transcription
    cleaned_transcription = transcription.replace("<s>", "").strip()
    print("Cleaned Transcription:", cleaned_transcription)

    # Save transcription to a text file
    output_dir = "./OUTPUT/TEXT"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "01_transcription.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_transcription)

    print(f"Transcription saved to: {output_file}")

    # Calculate WER
    with open(original_transcript_file, "r", encoding="utf-8") as f:
        original_transcript = f.read().strip()

    wer = jiwer.wer(original_transcript, cleaned_transcription)
    print(f"Word Error Rate (WER): {wer}")

except Exception as e:
    print(f"Error during inference: {e}")
