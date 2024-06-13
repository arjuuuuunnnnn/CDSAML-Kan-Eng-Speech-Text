# This script read a csv , for each row . it produces a audioin PWD . It dumps the audio into the dir 
import subprocess
import csv
import shutil 
shutil.rmtree('./temp_script.sh', ignore_errors=True)
shutil.rmtree('./*.wav', ignore_errors=True)
csv_file_path = './27FEBasr.csv'
op_dir = str("./27FEB/")
i = 0
with open(csv_file_path, 'r') as csv_file:

    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        i = i + 1
        text_to_synthesize = row[0]
        export_data = 'export data="{}"'.format(text_to_synthesize)       
        name_wav = str( op_dir +  str(i) + ".wav")
        export_name_wav = 'export name_of_wav="{}"'.format(name_wav)
        cmd = "python3 -m TTS.bin.synthesize --model_path ./models/v1/kn/fastpitch/best_model.pth --config_path ./models/v1/kn/fastpitch/config.json --vocoder_path ./models/v1/kn/hifigan/best_model.pth --vocoder_config_path ./models/v1/kn/hifigan/config.json --speaker_idx female  --out_path \"$name_of_wav\"  --text \"$data\""
        print(cmd)
        
        with open('temp_script.sh', 'w') as temp_script:
            temp_script.write(export_data)
            temp_script.write("\n")
            temp_script.write(export_name_wav)
            temp_script.write("\n")
            temp_script.write(cmd)
        try:
            subprocess.run(['bash', './temp_script.sh'], check=True)
            shutil.rmtree('./temp_script.sh', ignore_errors=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bash script: {e}")
