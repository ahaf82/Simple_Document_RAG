import openai
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import tkinter as tk
from tkinter import scrolledtext
import os
from dotenv import dotenv_values

# Function to generate initial text from the given script
def generate_initial_text(query_sentence):
    # Load the pre-trained model for generating the query embedding
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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

def summarize_text_stream(openai_api_key, text, output_text_widget, max_new_tokens=50, min_length=40):
    # Ensure the API key is set
    openai.api_key = openai_api_key

    # Initialize variables to store the stream content
    full_summary = ""

    messages = [
		{"role" : "system", "content" : "You got information from a vector database according pdf documents. You are building an understandable, factual text out of this information in german"},
		{"role" : "user", "content" : f"{text}"}
    ]
    
    # Make a request to the OpenAI API to summarize the text
    response = openai.chat.completions.create(
		model="gpt-3.5-turbo",
		messages=messages,
		max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.3,
        stream=True
	)

    print(response)
    # Process the stream
    for event in response:
        content_part = event.choices[0].delta.content
        output_text_widget.insert(tk.END, content_part)
        output_text_widget.see(tk.END)
        full_summary += content_part

    # Return the full summary
    return full_summary

def on_submit():
    query_sentence = query_entry.get()
    initial_text = generate_initial_text(query_sentence)
    summary_textbox.delete('1.0', tk.END)  # Clear previous summary
    summarize_text_stream(openai_api_key, initial_text, summary_textbox)

# Set up the UI
root = tk.Tk()
root.title("Dokumenten Suche und Auswertung")

# Summary output text box
summary_textbox = scrolledtext.ScrolledText(root, width=110, height=20)
summary_textbox.pack()

# Query input field
query_label = tk.Label(root, text="Deine Frage:")
query_label.pack()

query_entry = tk.Entry(root, width=100)
query_entry.pack()

# Submit button
submit_button = tk.Button(root, text="Abschicken", command=on_submit)
submit_button.pack()

# Example usage:
config = dotenv_values('.env')
openai_api_key = config["OPENAI_API_KEY"]  # Ensure this is set securely

root.mainloop()
