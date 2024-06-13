import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor

model_id = "facebook/mms-1b-all"
target_lang = "kan"

# Replace with your actual file path
audio_file = "audio.mp3"  # Make sure you have a file named "audio.wav" in the same directory

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
processor = AutoProcessor.from_pretrained(model_id, target_lang=target_lang)

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
    print("Transcription:", transcription)

except Exception as e:
    print(f"Error during inference: {e}")

