# import librosa
# import numpy as np
# import soundfile as sf

# def calculate_energy(y, frame_length=2048, hop_length=512):
#     energy = np.array([
#         sum(abs(y[i:i+frame_length]**2))
#         for i in range(0, len(y), hop_length)
#     ])
#     return energy

# def energy_based_split(y, sr, energy, threshold_ratio=0.01, frame_length=2048, hop_length=512):
#     threshold = np.max(energy) * threshold_ratio
#     silent_frames = np.where(energy < threshold)[0]
#     # Finding the boundaries of silent sections
#     boundaries = np.diff(silent_frames) > 1
#     start_indices = silent_frames[np.append([True], boundaries)]
#     end_indices = silent_frames[np.append(boundaries, [True])]
    
#     # Convert frame indices to samples
#     start_samples = start_indices * hop_length
#     end_samples = (end_indices * hop_length) + frame_length

#     segments = []
#     current_pos = 0
#     for start, end in zip(start_samples, end_samples):
#         if start > current_pos:
#             segments.append((current_pos, start))
#         current_pos = end
#     if current_pos < len(y):
#         segments.append((current_pos, len(y)))
#     return segments

# def split_audio_on_energy(audio_path, output_dir):
#     y, sr = librosa.load(audio_path, sr=None)
#     energy = calculate_energy(y)
#     segments = energy_based_split(y, sr, energy)

#     # Write segments to files
#     for i, (start, end) in enumerate(segments):
#         segment = y[start:end]
#         output_file_path = f"{output_dir}/segment_{i+1}.wav"
#         sf.write(output_file_path, segment, sr)
#         print(f"Saved segment: {output_file_path}")


# audio_path = 'C:/Users/Arvin/Desktop/0_SPEECH_TO_TEXT_TOOLS_KIT/audio.mp3'
# output_dir = 'C:/Users/Arvin/Desktop/0_SPEECH_TO_TEXT_TOOLS_KIT/segment'
# split_audio_on_energy(audio_path, output_dir)


import os
import sys
import numpy as np
import librosa
import soundfile as sf

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from src.SpeechText import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

os.makedirs("../../artifacts/STAGE_01", exist_ok=True)
STAGE_NAME = "Audio Splitting Stage"

class AudioSplitting:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

    def calculate_energy(self, y, frame_length=2048, hop_length=512):
        energy = np.array([
            sum(abs(y[i:i+frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        return energy

    def energy_based_split(self, y, sr, energy, threshold_ratio=0.01, frame_length=2048, hop_length=512):
        threshold = np.max(energy) * threshold_ratio
        silent_frames = np.where(energy < threshold)[0]
        boundaries = np.diff(silent_frames) > 1
        start_indices = silent_frames[np.append([True], boundaries)]
        end_indices = silent_frames[np.append(boundaries, [True])]
        
        start_samples = start_indices * hop_length
        end_samples = (end_indices * hop_length) + frame_length
        segments = []
        current_pos = 0
        for start, end in zip(start_samples, end_samples):
            if start > current_pos:
                segments.append((current_pos, start))
            current_pos = end
        if current_pos < len(y):
            segments.append((current_pos, len(y)))
        return segments

    def split_audio_on_energy(self):
        logger.info(f"Processing: {self.input_file}")
        y, sr = librosa.load(self.input_file, sr=None)
        energy = self.calculate_energy(y)
        segments = self.energy_based_split(y, sr, energy)
        
        os.makedirs(self.output_dir, exist_ok=True)
        for i, (start, end) in enumerate(segments):
            segment = y[start:end]
            output_file_path = os.path.join(self.output_dir, f"segment_{i+1}.wav")
            sf.write(output_file_path, segment, sr)
            logger.info(f"Saved segment: {output_file_path}")

    def main(self):
        self.split_audio_on_energy()

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        
        input_file = "../../artifacts/STAGE_00/Cleaned_denoiser_facebook_1_input.mp3"
        output_dir = "../../artifacts/STAGE_01"
        
        obj = AudioSplitting(input_file, output_dir)
        obj.main()
        
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(f"Error in pipeline execution: {str(e)}")
        sys.exit(1)
