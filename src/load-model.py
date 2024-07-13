# Load model directly
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, GPT2Tokenizer, GPT2LMHeadModel
from langchain_community.llms import huggingface_pipeline
from langchain.chains import RetrievalQA
import os

def is_model_cached(model_name, cache_dir):
    # Check for the presence of model files in the cache directory
    expected_model_files = ["pytorch_model.bin", "config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    for file_name in expected_model_files:
        if not os.path.exists(os.path.join(cache_dir, model_name.replace('/', '_'), file_name)):
            return False
    return True

def setup_model(model_name="distilgpt2", cache_dir="~/.cache/huggingface/transformers/"):
    # Expand the user directory
    cache_dir = os.path.expanduser(cache_dir)
    
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    if is_model_cached(model_name, cache_dir):
        print(f"Model '{model_name}' is already cached.")
    else:
        print(f"Model '{model_name}' is not cached. Downloading...")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        # Set the pad_token_id to avoid warnings
        model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        tokenizer.save_pretrained(f"./{model_name}")
        model.save_pretrained(f"./{model_name}")
        print("Model loaded successfully.")
    except OSError as e:
        print(f"Error loading model: {e}")
        tokenizer, model = None, None

    return tokenizer, model

def main():
    setup_model()
    
if __name__ == "__main__":
    main()