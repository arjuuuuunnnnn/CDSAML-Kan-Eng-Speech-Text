# from punctuators.models import PunctCapSegModelONNX
# from typing import List
# # Instantiate this model
# # This will download the ONNX and SPE models. To clean up, delete this model from your HF cache directory.
# m = PunctCapSegModelONNX.from_pretrained("pcs_47lang")

# # Define some input texts to punctuate
# input_texts: List[str] = [
#     "ಹೆಸರು ಕಂದ ನಾನು ಪಿಎಸ್ ಕಾಲೇಜಿನಲ್ಲಿ ಓದಿತ್ತಿದ್ದೇರೆ ನಾನು ಈಗ ಕಾಲೇಜಿನಲ್ಲಿದ್ದೇನೆ"
# ]
# results: List[List[str]] = m.infer(input_texts)
# for input_text, output_texts in zip(input_texts, results):
#     print(f"Input: {input_text}")
#     print(f"Outputs:")
#     for text in output_texts:
#         print(f"\t{text}")
#     print()












# from punctuators.models import PunctCapSegModelONNX
# from typing import List
# import os

# # Instantiate the model
# m = PunctCapSegModelONNX.from_pretrained("pcs_47lang")

# # Define some input texts to punctuate
# input_texts: List[str] = [
#     "ನಾನು ಹೆಸರು ಸಮೃದ್ ನನಗೆ ನಾಯಿ ಹಸು ಮತ್ತು ಬೆಕ್ಕು ಇಷ್ಟ ನಿನ್ನಗೆ ಹೆಸರೇನು ನಾನು ಪಿಎಸ್ ವಿಶ್ವವಿದ್ಯವಿದ್ಯಾಲಯದಲ್ಲಿ ಓದುತ್ತಿದ್ದೇನೆ"
# ]

# # Run inference
# results: List[List[str]] = m.infer(input_texts)

# # Define output directory
# output_dir = "./OUTPUT/TEXT"
# os.makedirs(output_dir, exist_ok=True)

# # Define output file path
# output_file = os.path.join(output_dir, "punctuation2.txt")

# # Write results to file
# with open(output_file, "w", encoding="utf-8") as f:
#     for input_text, output_texts in zip(input_texts, results):
#         f.write(f"Input: {input_text}\n")
#         f.write("Outputs:\n")
#         for text in output_texts:
#             f.write(f"\t{text}\n")
#         f.write("\n")

# print(f"Results saved to: {output_file}")






#-------------------------------------------------------------------------------------------------------------------------

from punctuators.models import PunctCapSegModelONNX
from typing import List
import os

# Instantiate the model
m = PunctCapSegModelONNX.from_pretrained("pcs_47lang")

# Define input file path
input_file = os.path.join("./OUTPUT/TEXT", "01_transcription.txt")

# Read input texts from the file
input_texts: List[str] = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        input_texts.append(line.strip())  # Remove leading/trailing whitespaces

# Run inference
results: List[List[str]] = m.infer(input_texts)

# Define output directory (same as before)
output_dir = "./OUTPUT/TEXT"
os.makedirs(output_dir, exist_ok=True)

# Define output file path (same as before)
output_file = os.path.join(output_dir, "02_punctuation.txt")

# Write results to file
with open(output_file, "w", encoding="utf-8") as f:
    for input_text, output_texts in zip(input_texts, results):
        f.write(f"Input: {input_text}\n")
        f.write("Outputs:\n")
        for text in output_texts:
            f.write(f"\t{text}\n")
        f.write("\n")

print(f"Results saved to: {output_file}")
