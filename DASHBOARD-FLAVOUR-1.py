#pip install inaSpeechSegmenter
#matplotlib
#scipy
#flask
#Ipython

import pandas as pd
import numpy as np
import subprocess
import os
import shutil
from io import StringIO

### ENABLE TO RUN inaSpeechSegmenter :-
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv


def delete_path(directory_path):
    try:
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Remove the directory and its contents recursively
            shutil.rmtree(directory_path)
            print(f"Directory deleted: {directory_path}")
        else:
            print(f"Directory not found: {directory_path}")
    except Exception as e:
        print(f"Error: {e}")




def force_delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file '{file_path}': {e}")





def get_mp3_files(directory):
    mp3_files = [file for file in os.listdir(directory) if file.endswith(".mp3")]
    return mp3_files

def move_file(src, dest):
    try:
        shutil.move(src, dest)
    except Exception as e:
        print(f"Error moving file from '{src}' to '{dest}': {e}")

def copy_and_rename_files(input_path, output_path):
    shutil.copy2(input_path, output_path)

def move_and_rename_mp3_files(input_directory, output_directory):
    mp3_files = [file for file in os.listdir(input_directory) if file.endswith(".mp3")]
    for mp3_file in mp3_files:
        os.makedirs(output_directory, exist_ok=True)
        input_path = os.path.join(input_directory, mp3_file)
        output_path = os.path.join(output_directory, mp3_file)
        os.rename(input_path, output_path)
        print(f"Moved and renamed: {mp3_file} -> {output_path}")

def process_audio(input_mp3s_directory, mp3_file_name):
    mp3_file = os.path.join(input_mp3s_directory, mp3_file_name)
    print("Now, Working On  --->  " + mp3_file)
####################################################
    # ENABLE THE BELOW FOR inaSpeechSegmenter MODEL :_
    seg = Segmenter()
    segmentation = seg(mp3_file)
    tmp_csv = os.path.join(INTERMITTENT_CSVS_WITH_AUDIO_CUT_TIME, f'{mp3_file_name}_MODEL_ANALYSIS.csv')
    seg2csv(segmentation, tmp_csv)
####################################################
    df = pd.read_csv(tmp_csv, delimiter='\t')
    df = df.sort_values(by='labels')
    numeric_array = df.iloc[:, 1:].values.astype(float)
    flat_numeric_array = np.concatenate(numeric_array)
    cut_times = np.unique(flat_numeric_array)
    segment_number = 0
    output_directory = os.path.join(BASE_PATH, "WORKING_FILES", "INTERMITTENT_CUT_AUDIOS")  # Updated to the desired output directory
    input_file_name = os.path.basename(mp3_file)
    for i in range(0, len(cut_times) - 1, 1):
        start_time = cut_times[i]
        stop_time = cut_times[i + 1] if i + 1 < len(cut_times) else cut_times[-1]
        print(start_time, stop_time)
        if start_time == stop_time:
            continue

        start_time_str = str(start_time)
        stop_time_str = str(stop_time)

        output_audio = f"CUT_{input_file_name}__{segment_number}__.mp3"

        command = [
            'ffmpeg',
            '-y',
            '-i', mp3_file,
            '-ss', start_time_str,
            '-to', stop_time_str,
            '-c:a', 'libmp3lame', '-q:a', '0',
            '-vn',
            '-c', 'copy',
            os.path.join(output_directory, output_audio)
        ]
        print(command)
        subprocess.run(command)
        segment_number += 1

    NEW_DIR = os.path.join(output_directory, f"{mp3_file_name}_CUTS")
    os.makedirs(NEW_DIR, exist_ok=True)
    move_file(tmp_csv, os.path.join(INTERMITTENT_CSVS_WITH_AUDIO_CUT_TIME, f'{mp3_file_name}_MODEL_ANALYSIS.csv'))
    input_mp3s_directory = output_directory
    move_and_rename_mp3_files(input_mp3s_directory, NEW_DIR)
    input_mp3s_directory = INPUT_MP3S
    mp3_file = os.path.join(input_mp3s_directory, mp3_file_name)
    html_input_file = "MAKE_HTML.py"
    html_input_file = os.path.join(input_mp3s_directory, html_input_file)
    copy_and_rename_files(mp3_file, os.path.join(NEW_DIR, f"ORIGINAL__{mp3_file_name}"))
    copy_and_rename_files(html_input_file, os.path.join(NEW_DIR))


BASE_PATH = str(".")
INPUT_MP3S = os.path.join(BASE_PATH, "INPUT_MP3S")
WORKING_DIRECTORY = os.path.join(BASE_PATH, "WORKING_FILES")
INTERMITTENT_CSVS_WITH_AUDIO_CUT_TIME = os.path.join(WORKING_DIRECTORY, "INTERMITTENT_CSVS_WITH_AUDIO_CUT_TIME")
INTERMITTENT_CUT_AUDIOS = os.path.join(WORKING_DIRECTORY, "INTERMITTENT_CUT_AUDIOS")
delete_path(INTERMITTENT_CSVS_WITH_AUDIO_CUT_TIME)
delete_path(INTERMITTENT_CUT_AUDIOS)

os.makedirs(INPUT_MP3S, exist_ok=True)
os.makedirs(INTERMITTENT_CSVS_WITH_AUDIO_CUT_TIME, exist_ok=True)
os.makedirs(INTERMITTENT_CUT_AUDIOS, exist_ok=True)


mp3_files = get_mp3_files(INPUT_MP3S)
for mp3_file in mp3_files:
    process_audio(INPUT_MP3S, mp3_file)