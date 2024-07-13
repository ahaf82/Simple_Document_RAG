import openai
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
from dotenv import dotenv_values


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
    search_params = {"metric_type": "L2", "params": {"nprobe": 5}}

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

def summarize_text(openai_api_key, text):
    # Ensure the API key is set
    openai.api_key = openai_api_key

    messages = [
		{"role" : "system", "content" : "You got information from a vector database according pdf documents. You are building an understandable, factual text out of this information in german"},
		{"role" : "user", "content" : f"{text}"}
    ]
    
    # Make a request to the OpenAI API to summarize the text
    response = openai.chat.completions.create(
		model="gpt-4",
		messages=messages,
		max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.3
	)

    summary = response.choices[0].message.content
    return summary

# Example usage:
if __name__ == "__main__":
    config = dotenv_values('.env')
    openai_api_key = config["OPENAI_API_KEY"]
    initial_text = generate_initial_text()
    summary = summarize_text(openai_api_key, initial_text)
    print(summary)
