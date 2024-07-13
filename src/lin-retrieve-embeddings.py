from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, pipeline, TRANSFORMERS_CACHE, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
import sentencepiece as spm
import torch
import os
from langchain_community.llms import huggingface_pipeline
from langchain.chains import RetrievalQA

# Function to generate initial text from the given script
def generate_initial_text():
    # Load the pre-trained model for generating the query embedding
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Sample query sentence
    query_sentence = "Was darf man mit SMCB signieren"

    # Generate the query embedding
    query_embedding = model.encode([query_sentence])

    # Connect to Milvus server
    connections.connect(alias="default", host="localhost", port="19530")

    # Specify the collection name where embeddings are stored
    collection_name = "pdf_collection"

    # Load the existing collection
    collection = Collection(name=collection_name)

    # Load the collection for searching
    collection.load()

    # Define search parameters
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    # Perform the search
    results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["content"]  # Ensure the 'content' field is retrieved
    )

    # Concatenate results into a single string
    initial_text = extract_search_results_content(results)
    return initial_text

# Function to extract the 'content' field from entity
# Function to extract content from search results
def extract_search_results_content(results):
    content_list = []
    for result in results:
        for hit in result:
            # Adjust this part based on the actual structure of the hit object
            # Assuming hit is a dictionary-like object or has attributes directly accessible
            if isinstance(hit, dict):
                content = hit.get("content")
            else:
                content = getattr(hit, "content", None)
            if content:
                content_list.append(content)
    return ' '.join(content_list)


def is_model_cached(model_name, cache_dir):
    # Check for the presence of model files in the cache directory
    expected_model_files = ["pytorch_model.bin", "config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    for file_name in expected_model_files:
        if not os.path.exists(os.path.join(cache_dir, model_name.replace('/', '_'), file_name)):
            return False
    return True

def setup_model(model_name="openlm-research/open_llama_3b", cache_dir="~/.cache/huggingface/transformers/"):
    # Expand the user directory
    cache_dir = os.path.expanduser(cache_dir)
    
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    if is_model_cached(model_name, cache_dir):
        print(f"Model '{model_name}' is already cached.")
    else:
        print(f"Model '{model_name}' is not cached. Downloading...")

    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set the pad_token_id to avoid warnings
        # Set the pad_token_id to eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        tokenizer.save_pretrained(f"./{model_name}")
        model.save_pretrained(f"./{model_name}")
        print("Model loaded successfully.")
    except OSError as e:
        print(f"Error loading model: {e}")
        tokenizer, model = None, None

    return tokenizer, model


def summarize_text(tokenizer, model, text, max_new_tokens=50, min_length=40, length_penalty=2.0, num_beams=4):
    max_length = 2048  # typically 2048 for LLaMA models
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=max_length - max_new_tokens, truncation=True)

    # Ensure input length is within the model's max length
    if inputs.shape[1] > max_length - max_new_tokens:
        inputs = inputs[:, :max_length - max_new_tokens]

    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    # Step 1: Generate initial text
    initial_text = generate_initial_text()
    print("Initial Text:", initial_text)
    
    # Step 2: Download and set up Llama 3 8B model
    tokenizer, model = setup_model()
    if tokenizer is not None and model is not None:
        print("Model setup is complete.")
    else:
        print("Model setup failed.")    

    # Step 3: Enhance initial text
    generated_text = summarize_text(tokenizer, model, initial_text)
    print("Enhanced Text:", generated_text)

if __name__ == "__main__":
    main()
