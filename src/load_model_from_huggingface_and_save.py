from transformers import RobertaTokenizer, RobertaModel
import os

# Specify the directory to save the model
save_directory = "./codebert_model"

# Load the pre-trained CodeBERT tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

# Save the tokenizer and model locally
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)