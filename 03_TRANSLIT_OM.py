# from om_transliterator import Transliterator
# transliterator = Transliterator()
# original_text = "ನೀ ಮಾಯೆಯೊಳಗೋ ನಿನ್ನೊಳು ಮಾಯೆಯೋ"
# transliterated_text = transliterator.knda_to_latn(original_text)
# print(transliterated_text)




# from om_transliterator import Transliterator

# # Initialize the transliterator
# transliterator = Transliterator()

# # Define file paths
# input_file_path = 'OUTPUT/TEXT/punctuation2.txt'
# output_file_path = 'OUTPUT/TEXT/transliteration.txt'

# # Read the original text from the file
# with open(input_file_path, 'r', encoding='utf-8') as file:
#     original_text = file.read()

# # Perform transliteration
# transliterated_text = transliterator.knda_to_latn(original_text)

# # Write the transliterated text to the output file
# with open(output_file_path, 'w', encoding='utf-8') as file:
#     file.write(transliterated_text)







































#-------------------------------------------------------------------------------------------------------------------------

from om_transliterator import Transliterator

# Initialize the transliterator
transliterator = Transliterator()

# Define file paths
input_file_path = 'OUTPUT/TEXT/02_punctuation.txt'
output_file_path = 'OUTPUT/TEXT/03_transliteration.txt'

# Read the original text from the file
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Find the line with "Outputs:" and read the subsequent lines
start_reading = False
original_text = ""
for line in lines:
    if start_reading:
        original_text += line.strip() + " "
    if "Outputs:" in line:
        start_reading = True

# Perform transliteration
transliterated_text = transliterator.knda_to_latn(original_text.strip())

# Write the transliterated text to the output file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(transliterated_text)
