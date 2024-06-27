# import torch
# from transformers import (
#     AutoModelForSeq2SeqLM,
#     AutoTokenizer,
# )
# from IndicTransTokenizer import IndicProcessor


# model_name = "ai4bharat/indictrans2-indic-en-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

# ip = IndicProcessor(inference=True)

# input_sentences = [
#     "ನಾನು ಹೆಸರು ಸಮೃದ್, ನನಗೆ ನಾಯಿ ಹಸು ಮತ್ತು ಬೆಕ್ಕು ಇಷ್ಟ. ನಿನ್ನಗೆ ಹೆಸರೇನು? ನಾನು ಪಿಎಸ್ ವಿಶ್ವವಿದ್ಯವಿದ್ಯಾಲಯದಲ್ಲಿ ಓದುತ್ತಿದ್ದೇನೆ.",
# ]

# src_lang, tgt_lang = "kan_Knda", "eng_Latn"

# batch = ip.preprocess_batch(
#     input_sentences,
#     src_lang=src_lang,
#     tgt_lang=tgt_lang,
# )

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Tokenize the sentences and generate input encodings
# inputs = tokenizer(
#     batch,
#     truncation=True,
#     padding="longest",
#     return_tensors="pt",
#     return_attention_mask=True,
# ).to(DEVICE)

# # Generate translations using the model
# with torch.no_grad():
#     generated_tokens = model.generate(
#         inputs,
#         use_cache=True,
#         min_length=0,
#         max_length=256,
#         num_beams=5,
#         num_return_sequences=1,
#     )

# # Decode the generated tokens into text
# with tokenizer.as_target_tokenizer():
#     generated_tokens = tokenizer.batch_decode(
#         generated_tokens.detach().cpu().tolist(),
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=True,
#     )

# # Postprocess the translations, including entity replacement
# translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

# for input_sentence, translation in zip(input_sentences, translations):
#     print(f"{src_lang}: {input_sentence}")
#     print(f"{tgt_lang}: {translation}")




#-------------------------------------------------------------------------------------------------------------------------

import torch
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransTokenizer import IndicProcessor


model_name = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

ip = IndicProcessor(inference=True)

# Define input file path
input_file = os.path.join("./OUTPUT/TEXT", "02_punctuation.txt")

# Define output file path
output_file = os.path.join("./OUTPUT/TEXT", "04_translation.txt")

# Read input texts from the file
input_texts = []
current_input_text = ""
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip().lower().startswith("outputs:"):
            if current_input_text:
                input_texts.append(current_input_text.strip())
            current_input_text = ""
        else:
            current_input_text += line

    if current_input_text:
        input_texts.append(current_input_text.strip())

# Check if any input texts were found
if not input_texts:
    print("No input texts found in the file.")
    exit()

src_lang, tgt_lang = "kan_Knda", "eng_Latn"

# Process input texts in batches
batch_size = 8  # Batch size for processing input texts
for i in range(0, len(input_texts), batch_size):
    batch_input_texts = input_texts[i:i + batch_size]

    # Preprocess the batch
    batch = ip.preprocess_batch(batch_input_texts, src_lang=src_lang, tgt_lang=tgt_lang)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize the sentences and generate input encodings
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess the translations, including entity replacement
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    # Write translations to output file
    with open(output_file, "a", encoding="utf-8") as f:
        for translation in translations:
            if not translation.startswith(f"{src_lang}:") and not translation.startswith(f"{tgt_lang}:"):
                f.write(f"Output: {translation}\n")

print(f"Translations saved to: {output_file}")
