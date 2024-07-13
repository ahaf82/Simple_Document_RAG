# RAG_Tests

### Install python packages
pip install -r requirements.txt

### Download and run Milvus Vector DB
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start

### Embed PDF documents in embedding_documents folder
python src/embed-documents-in-vectordb.py

### Retrieve Information from Promptext in retrieval file and test with local LLM
python src/win-retrieve-embeddings.py