# RAG_Tests

### Download and run Milvus Vector DB in docker container
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start

### Create python environement and install python packages
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt

### Embed PDF documents in embedding_documents folder 
you can add your own ones - the ones here are from the german gematik GmbH
**python src/embed-documents-in-vectordb.py**

### Retrieve Information from Promptext in retrieval file and test with local LLM
python src/win-retrieve-embeddings.py

### Use Mini UI with OpenAI API to get summarized information from documents
create *.env* file with your openAI key (OPENAI_API_KEY=<your key>) 
open ui with **src/ui-lin-retrieve-embeddings_from_openai_api.py**
