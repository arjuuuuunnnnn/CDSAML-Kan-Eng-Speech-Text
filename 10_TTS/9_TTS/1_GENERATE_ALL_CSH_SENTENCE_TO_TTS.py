############
csv_file_name = '27FEBasr.csv'
root = "./"
csv_path = root + csv_file_name
OUTPUT_DIR = "./NEW/"
######
import subprocess
import csv
import shutil 
shutil.rmtree('./temp_script.sh', ignore_errors=True)
shutil.rmtree('./*.wav', ignore_errors=True)
def add_line_to_script(script_path, line):
    sed_command = f"sed -i '1i\\{line}' {script_path}"
    subprocess.run(sed_command, shell=True)
i = 0
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        i = i + 1
        text_to_synthesize = row[0]
        export_data = 'export data="{}"'.format(text_to_synthesize)
        name_wav = str( OUTPUT_DIR + str(i)+".wav")
        export_name_wav = 'export name_of_wav="{}"'.format(name_wav)
        cmd = "python3 -m TTS.bin.synthesize --model_path ./models/v1/kn/fastpitch/best_model.pth --config_path ./models/v1/kn/fastpitch/config.json --vocoder_path ./models/v1/kn/hifigan/best_model.pth --vocoder_config_path ./models/v1/kn/hifigan/config.json --speaker_idx female --use_cuda 1 --out_path \"$name_of_wav\"  --text \"$data\""
        print(cmd)
        export_name_wav_clear = 'export name_of_wav=\"\"'
        export_data_clear = 'export data=\"\"' 
        filename = csv_path + "-TTS-CMDS.csh" 
        with open( filename  , 'a+') as temp_script:
            temp_script.write(export_data)
            temp_script.write("\n")
            temp_script.write(export_name_wav)
            temp_script.write("\n")
            temp_script.write(cmd)
            temp_script.write("\n")   
            temp_script.write(export_name_wav_clear)
            temp_script.write("\n")   
            temp_script.write(export_data_clear)
            temp_script.write("\n")




line_to_add = "conda activate tts-env"
add_line_to_script(filename, line_to_add)
