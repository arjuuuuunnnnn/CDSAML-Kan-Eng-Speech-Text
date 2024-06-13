########################################################
# NOTE ONLY 16KHz Audio Files can be  transcribed .
########################################################
import os
import librosa
import IPython.display as ipd
import nemo
import nemo.collections.asr as nemo_asr
quartznet = nemo_asr.models.ASRModel.from_pretrained(model_name="QuartzNet15x5Base-En")
data_dir = "./"
files = [os.path.join(data_dir, 'TEST_16KHZ.mp3')]
for fname, transcription in zip(files, quartznet.transcribe(audio=files)):
  print(f"Audio in {fname} was recognized as: {transcription}")