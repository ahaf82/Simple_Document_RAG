import openai
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import tkinter as tk
from tkinter import scrolledtext
import os
from dotenv import dotenv_values

initial_message_content = [
    {"role" : "system", "content" : "You got information from a vector database according pdf documents. You are building an understandable, factual text out of this information in german"},
]
messages = initial_message_content

# Function to generate initial text from the given script
def generate_initial_text(query_sentence):
    # Load the pre-trained model for generating the query embedding
    model = SentenceTransformer('./all-mpnet-base-v2')

    # Generate the query embedding
    query_embedding = model.encode([query_sentence])

    # Connect to Milvus server
    connections.connect(alias="default", host="localhost", port="19530")

    # Specify the collection name where embeddings are stored
    collection_name = "document_collection"

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
    initial_text = f"""
    The initial question is: {query_sentence}
    The context retrieved from the vector db is: {extract_search_results_content(results)}
    """
    
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

    messages.append({"role": "user", "content": text})
    output_text_widget.insert(tk.END, f"Antwort: \r\n\n")
    
    # Make a request to the OpenAI API to summarize the text
    response = openai.chat.completions.create(
		model="gpt-4o-mini",
		messages=messages,
		max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.3,
        stream=True
	)

    # Process the stream
    for event in response:
        content_part = event.choices[0].delta.content
        if content_part is not None:
            output_text_widget.insert(tk.END, content_part)
            output_text_widget.see(tk.END)
            full_summary += content_part

    # Return the full summary
    output_text_widget.insert(tk.END, "\n\r\n")
    messages.append({"role": "user", "content": full_summary})
    return full_summary

def on_submit(event=None):
    query_sentence = query_entry.get()
    query_entry.delete(0, tk.END)
    initial_text = generate_initial_text(query_sentence)
    summary_textbox.insert(tk.END, f"Frage: {query_sentence}\r\n\n")
    summarize_text_stream(openai_api_key, initial_text, summary_textbox)

def on_new_question():
    messages = initial_message_content
    summary_textbox.delete('1.0', tk.END)  # Clear previous summary
    query_entry.delete(0, tk.END)

# Set up the UI
root = tk.Tk()
root.title("Dokumenten Suche und Auswertung")

# Summary output text box
summary_textbox = scrolledtext.ScrolledText(root, width=110, height=35)
summary_textbox.pack()

# Query input field
query_label = tk.Label(root, text="Deine Frage:")
query_label.pack()

query_entry = tk.Entry(root, width=100)
query_entry.pack()
query_entry.bind("<Return>", on_submit)  # Bind the Enter key to the on_submit function

# Button frame
button_frame = tk.Frame(root)
button_frame.pack()

# Submit button
submit_button = tk.Button(button_frame, text="Abschicken", command=on_submit)
submit_button.pack(side=tk.LEFT)

# New question button
new_question_button = tk.Button(button_frame, text="Neue Frage", command=on_new_question)
new_question_button.pack(side=tk.LEFT)

# Example usage:
config = dotenv_values('.env')
openai_api_key = config["OPENAI_API_KEY"]  # Ensure this is set securely

root.mainloop()

# import openai
# from transformers import GPT2Tokenizer, GPT2Model
# import torch
# from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema
# import tkinter as tk
# from tkinter import scrolledtext
# from dotenv import dotenv_values

# # Load the API key from the environment
# config = dotenv_values('.env')
# openai_api_key = config["OPENAI_API_KEY"]

# # Initialize the initial message content
# initial_message_content = [
#     {"role": "system", "content": "You retrieved information from a vector database according to PDF documents. You are building an understandable, factual text out of this information in German."},
# ]
# messages = initial_message_content.copy()

# # Connect to Milvus server
# connections.connect(alias="default", host="localhost", port="19530")

# # Specify the collection name where embeddings are stored
# collection_name = "pdf_collection"

# # Load the existing collection
# collection = Collection(name=collection_name)

# # Initialize the GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2Model.from_pretrained("gpt2")

# def generate_gpt2_embedding(text):
#     """Generate embeddings using GPT-2 model."""
#     inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# def generate_initial_text(query_sentence):
#     """Generate initial text from the given query sentence."""
#     # Embed the query sentence using GPT-2
#     query_embedding = generate_gpt2_embedding(query_sentence)

#     # Perform the search using Milvus
#     search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
#     results = collection.search(
#         data=[query_embedding.tolist()],
#         anns_field="embedding",
#         param=search_params,
#         limit=3,
#         output_fields=["content"]
#     )

#     # Concatenate results into a single string
#     initial_text = f"""
#     The initial question is: {query_sentence}
#     The context retrieved from the vector db is: {extract_search_results_content(results)}
#     """

#     return initial_text

# def extract_search_results_content(results):
#     """Extract the 'content' field from search results."""
#     content_list = []
#     for result in results:
#         for hit in result:
#             content = hit.entity.get("content", "")
#             if content:
#                 content_list.append(content)
#     return ' '.join(content_list)

# def summarize_text_stream(api_key, text, output_widget, max_tokens=2000, temperature=0.3):
#     """Summarize the text using OpenAI's API with streaming."""
#     # Set the API key for OpenAI
#     openai.api_key = api_key

#     # Append the user's message
#     messages.append({"role": "user", "content": text})
#     output_widget.insert(tk.END, "Antwort:\r\n\n")

#     # Make a request to the OpenAI API to summarize the text
#     response = openai.ChatCompletion.create(
#         model="gpt-2",  # Using GPT-2
#         messages=messages,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         stream=True
#     )

#     # Process the stream and display the result in the text widget
#     full_summary = ""
#     for event in response:
#         content_part = event.choices[0].delta.get("content", "")
#         output_widget.insert(tk.END, content_part)
#         output_widget.see(tk.END)
#         full_summary += content_part

#     # Finalize the message content
#     output_widget.insert(tk.END, "\n\r\n")
#     messages.append({"role": "user", "content": full_summary})
#     return full_summary

# def on_submit(event=None):
#     """Handle the submit button click or Enter key press."""
#     query_sentence = query_entry.get()
#     query_entry.delete(0, tk.END)
#     initial_text = generate_initial_text(query_sentence)
#     summary_textbox.insert(tk.END, f"Frage: {query_sentence}\r\n\n")
#     summarize_text_stream(openai_api_key, initial_text, summary_textbox)

# def on_new_question():
#     """Reset the interface for a new question."""
#     global messages
#     messages = initial_message_content.copy()  # Reset the message list
#     summary_textbox.delete('1.0', tk.END)  # Clear previous summary
#     query_entry.delete(0, tk.END)

# # Set up the UI
# root = tk.Tk()
# root.title("Dokumenten Suche und Auswertung")

# # Summary output text box
# summary_textbox = scrolledtext.ScrolledText(root, width=110, height=35)
# summary_textbox.pack()

# # Query input field
# query_label = tk.Label(root, text="Deine Frage:")
# query_label.pack()

# query_entry = tk.Entry(root, width=100)
# query_entry.pack()
# query_entry.bind("<Return>", on_submit)  # Bind the Enter key to the on_submit function

# # Button frame
# button_frame = tk.Frame(root)
# button_frame.pack()

# # Submit button
# submit_button = tk.Button(button_frame, text="Abschicken", command=on_submit)
# submit_button.pack(side=tk.LEFT)

# # New question button
# new_question_button = tk.Button(button_frame, text="Neue Frage", command=on_new_question)
# new_question_button.pack(side=tk.LEFT)

# root.mainloop()
