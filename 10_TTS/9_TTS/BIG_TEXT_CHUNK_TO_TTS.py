#160 Words is the max for the this TTS . Else you hit Errors 

import subprocess
file_path = './KANNADA_TEXT.txt'
with open(file_path, 'r') as file:
    file_data = file.read()
    export_data = 'export data="{}"'.format(file_data)
    cmd = "python3 -m TTS.bin.synthesize  --model_path models/v1/kn/fastpitch/best_model.pth     --config_path models/v1/kn/fastpitch/config.json      --vocoder_path models/v1/kn/hifigan/best_model.pth     --vocoder_config_path models/v1/kn/hifigan/config.json  --speaker_idx female --use_cuda 1   --out_path  ./speech.wav  --text \"$data\"  "
    print(cmd)
    with open('temp_script.sh', 'w') as temp_script:
        temp_script.write(export_data)
        temp_script.write("\n")
        temp_script.write(cmd)

try:
    subprocess.run(['bash', './temp_script.sh'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running Bash script: {e}")
#finally:
#    subprocess.run(['rm11', 'temp_script.sh'])
