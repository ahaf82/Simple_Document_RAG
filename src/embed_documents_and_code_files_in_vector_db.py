import os
import xml.etree.ElementTree as ET
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import true
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Tokenizer
import torch
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# Initialize CodeBERT
codebert_model = AutoModel.from_pretrained("./codebert_model")
codebert_tokenizer = AutoTokenizer.from_pretrained("./codebert_model")

# Initialize the HuggingFace Embedding with GPT-2
embed_model = HuggingFaceEmbedding(model_name="./all-mpnet-base-v2")
Settings.embed_model = embed_model

def chunk_text(text, max_length=500):
    # Split text into chunks of max_length
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])

def embed_text_with_codebert(text):
    inputs = codebert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = codebert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def drop_collection(collection_name, all_collections):
        # Check if the collection exists
    if collection_name in all_collections:
        # Load the collection
        collection = Collection(name=collection_name)
        # Drop the collection
        collection.drop()
        print(f"Collection '{collection_name}' has been deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")


def process_files_in_directory(directory_path):
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Verify the connection
    connected = connections.has_connection("default")
    print(f"Connected to Milvus: {connected}")

    collections = ["document_collection", "xml_embeddings","pdf_collection"]

    # List all collections
    all_collections = utility.list_collections()
    print("\nCollections:", all_collections)
    
    for collection in collections:
        drop_collection(collection, all_collections)

    # Define the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Adjust dim based on the model output
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "Example collection")

    # Create a collection
    collection = Collection(collections[0], schema)

    # Use SimpleDirectoryReader to recursively read files and embed them
    reader = SimpleDirectoryReader(directory_path, recursive=true)
    documents = reader.load_data()

    # Process each document
    for doc in documents:
        # Extract the file extension from metadata or filename
        file_extension = doc.metadata.get('file_extension', None)
        if not file_extension:
            # Fallback to extracting the file extension from the filename
            filename = doc.metadata.get('file_name', '')
            file_extension = os.path.splitext(filename)[1][1:].lower()  # Get the extension without the dot

        # Chunk text if it's too long
        chunks = list(chunk_text(doc.text))

        for chunk in chunks:
            # If it's a Java file, use CodeBERT, otherwise use HuggingFace embeddings
            if file_extension == 'java':
                text_vector = embed_text_with_codebert(chunk)  # Use CodeBERT for Java files
            else:
                # Get embeddings using the HuggingFaceEmbedding model
                text_vector = embed_model._embed(chunk)  # Embed returns a list of embeddings

            # Debugging: Print shapes and types before inserting
            # print(f"Embedding: {text_vector}")
            print(f"Chunk: {chunk}")

            # Insert vector and content into Milvus
            collection.insert([{"embedding": text_vector, "content": chunk}])

    # Create an index for the 'embedding' field in Milvus
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

def main():
    directory_path = "embedding_documents_and_code"
    process_files_in_directory(directory_path)

if __name__ == "__main__":
    main()

