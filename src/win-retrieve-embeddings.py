from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, TRANSFORMERS_CACHE
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
    query_sentence = "Wie funktionier der Signaturservice"

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
    return content_list


def is_model_cached(model_name):
    # Define the cache directory explicitly
    cache_dir = os.path.join(os.path.expanduser("~/.cache/huggingface/transformers/"), model_name.replace('/', '_'))
    # Check for the presence of model files
    expected_model_files = ["pytorch_model.bin", "config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    for file_name in expected_model_files:
        if not os.path.exists(os.path.join(cache_dir, file_name)):
            return False
    return True

def setup_model(model_name="chavinlo/alpaca-native"):
    if is_model_cached(model_name):
        print(f"Model '{model_name}' is already cached.")
    else:
        print(f"Model '{model_name}' is not cached. Downloading...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.expanduser("~/.cache/huggingface/transformers/"))
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.path.expanduser("~/.cache/huggingface/transformers/"))
        print("Model loaded successfully.")
    except OSError as e:
        print(f"Error loading model: {e}")
        tokenizer, model = None, None

    return tokenizer, model

# Function to enhance text using Llama 3 8B
def enhance_text(initial_text, tokenizer, model):
    inputs = tokenizer(initial_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    enhanced_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return enhanced_text

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
    enhanced_text = enhance_text(initial_text, tokenizer, model)
    print("Enhanced Text:", enhanced_text)



if __name__ == "__main__":
    main()
