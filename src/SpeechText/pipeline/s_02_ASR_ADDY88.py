import os
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from src.SpeechText import logger

STAGE_NAME = "Audio Transcription Stage"
os.makedirs("../../artifacts/STAGE_02", exist_ok=True)

class AudioTranscription:
    def __init__(self, audio_file, output_dir, model_id):
        self.audio_file = audio_file
        self.output_dir = output_dir
        self.model_id = model_id

    def load_audio(self):
        try:
            if self.audio_file.lower().endswith('.mp3'):
                audio, orig_freq = torchaudio.backend.sox_io_backend.load(self.audio_file)
            else:
                audio, orig_freq = torchaudio.load(self.audio_file)
            audio = audio.mean(dim=0, keepdim=True)
            resampler = torchaudio.transforms.Resample(orig_freq, 16_000)
            audio = resampler(audio)
            return {"input_values": audio}
        except OSError as e:
            logger.error(f"Error loading audio file: {e}")
            raise

    def transcribe_audio(self, audio_input):
        processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        try:
            inputs = processor(
                audio_input["input_values"].numpy(),
                return_tensors="pt",
                sampling_rate=16_000
            )
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            return transcription
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def clean_transcription(self, transcription):
        return transcription.replace("<s>", "").strip()

    def save_transcription(self, transcription):
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, "02_transcription.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        logger.info(f"Transcription saved to: {output_file}")

    def main(self):
        audio_input = self.load_audio()
        transcription = self.transcribe_audio(audio_input)
        logger.info(f"Original Transcription: {transcription}")
        cleaned_transcription = self.clean_transcription(transcription)
        logger.info(f"Cleaned Transcription: {cleaned_transcription}")
        self.save_transcription(cleaned_transcription)

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        audio_file = "../../artifacts/STAGE_00/Cleaned_denoiser_facebook_1_input.mp3"
        output_dir = "../../artifacts/STAGE_02"
        model_id = "addy88/wav2vec2-kannada-stt"
        obj = AudioTranscription(audio_file, output_dir, model_id)
        obj.main()
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(e)
        raise e

