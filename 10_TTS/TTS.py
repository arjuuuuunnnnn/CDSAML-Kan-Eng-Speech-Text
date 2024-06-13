##USAGE:-
################################
#python TTS.py --input "ಕನ್ನಡದಲ್ಲಿ ನಮ್ಮ ಭಾಷೆ ಬಹು ಸುಂದರವಾದುದು."  --output kan.wav --language "Kannada"
#python TTS.py --input_text "Hi,How are you?"  --output eng.wav --language "English"
################################
import os
import torch
import torchaudio
from transformers import VitsModel, AutoTokenizer
def synthesize_speech(language, text, output_path):
    """
    Synthesizes speech from a given text string in the specified language.
        language (str): The language code (e.g., "Kannada").
        text (str): The text string to be converted to speech.
        output_path (str): The path to save the generated audio file (WAV format).
    """
    if language == "Kannada":
        model_name = "facebook/mms-tts-kan"
    elif language == "English":
        model_name = "facebook/mms-tts-eng"            
    else:
        raise ValueError(f"Unsupported language: {language}")
    if os.path.exists(output_path):  # Check if the file already exists
        os.remove(output_path)  # Force remove the existing file
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    torchaudio.save(output_path, output, sample_rate=model.config.sampling_rate)
    print(f"Speech synthesized and saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Text-to-Speech Synthesizer")
    parser.add_argument("--language", required=True, help="Language code (e.g., Kannada)")
    parser.add_argument("--input_text", required=True, help="Text string to convert to speech")
    parser.add_argument("--output", required=True, help="Path to save the output WAV file")
    args = parser.parse_args()
    synthesize_speech(args.language, args.input_text, args.output)