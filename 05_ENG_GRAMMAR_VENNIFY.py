# from transformers import pipeline

# def grammar_check(text):
#     # Initialize the text2text-generation pipeline with the grammar correction model
#     corrector = pipeline("text2text-generation", model="hafidikhsan/happy-transformer-t5-base-grammar-correction-lr-v1")

#     # Generate the corrected text
#     corrected = corrector(text, max_length=len(text) + 50)[0]['generated_text']

#     return corrected

# # Example usage
# input_text = "My name is Samrid, I like dog, cow and cat. What's your name? I'm studying at PS University."
# corrected_text = grammar_check(input_text)
# print(f"Original: {input_text}")
# print(f"Corrected: {corrected_text}")



# from transformers import pipeline

# def grammar_check(input_file, output_file):
#     # Initialize the text2text-generation pipeline with the grammar correction model
#     corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

#     # Read input text from file
#     with open(input_file, 'r', encoding='utf-8') as file:
#         input_text = file.read()

#     # Generate the corrected text
#     corrected = corrector(input_text, max_length=len(input_text) + 50)[0]['generated_text']

#     # Write corrected text to output file
#     with open(output_file, 'w', encoding='utf-8') as file:
#         file.write(corrected)

#     print(f"Grammar check completed. Corrected text saved to {output_file}")

# # Example usage
# input_file = "OUTPUT\TEXT\output.txt"  # Replace with your input file name
# output_file = "OUTPUT\TEXT\corrected_output.txt"  # Replace with your desired output file name

# grammar_check(input_file, output_file)
























#-------------------------------------------------------------------------------------------------------------------------

from transformers import pipeline

def grammar_check(text):
    # Initialize the text2text-generation pipeline with the grammar correction model
    corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
    
    # Generate the corrected text
    corrected = corrector(text, max_length=len(text) + 50)[0]['generated_text']
    
    return corrected

def get_text_from_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Find the second occurrence of "Output:"
    second_output_index = content.find("Output:", content.find("Output:") + 1)
    if second_output_index == -1:
        raise ValueError("The file does not contain a second 'Output:'")

    # Extract text after the second "Output:"
    text_to_correct = content[second_output_index + len("Output:"):].strip()
    
    return text_to_correct

def write_text_to_file(filepath, text):
    with open(filepath, 'w') as file:
        file.write(text)

# Example usage
file_path = "./OUTPUT/TEXT/04_translation.txt"
output_path = "./OUTPUT/TEXT/05_english_grammar.txt"

input_text = get_text_from_file(file_path)
corrected_text = grammar_check(input_text)

write_text_to_file(output_path, "Output: " + corrected_text)

print(f"Original: {input_text}")
print(f"Corrected: {corrected_text}")
