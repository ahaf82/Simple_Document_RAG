# Simple_RAG_System
<br>In this little project, I'm taking my first steps in creating a simple RAG implementation using Python, Milvus Vector DB, and ChatGPT. At the moment, it can embed PDF, DOCX, XLSX, and XML files in the vector database to use them in the RAG system.</br> 
<br>The main purpose is to learn how to set up a RAG system with a really simple UI in python and test different transformer models, LLM APIs and how the change of prompts or other configuration properties effect the output.</br>

### Download and run Milvus Vector DB in docker container
<br>curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh</br>
<br>bash standalone_embed.sh start</br>

### Create python environement and install python packages
<br>python -m venv ./venv</br>
<br>source ./venv/bin/activate</br>
<br>pip install -r requirements.txt</br>

### Embed PDF documents in embedding_documents folder 
<br>you can add your own ones - the ones here are from the german gematik GmbH</br>
<br>**python src/embed-documents-in-vectordb.py**</br>

### Retrieve Information from Promptext in retrieval file and test with local LLM
<br>python src/win-retrieve-embeddings.py</br>

### Use Mini UI with OpenAI API to get summarized information from documents
<br>create *.env* file with your openAI key (OPENAI_API_KEY=\<your_key\>)</br>
<br>open ui with **src/ui-lin-retrieve-embeddings_from_openai_api.py**</br>

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
