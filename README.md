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

### Third-Party Licenses

This project uses the following third-party libraries:

| Name                      | Version     | License                                            |
|---------------------------|-------------|----------------------------------------------------|
| Jinja2                    | 3.1.4       | BSD License                                        |
| MarkupSafe                | 2.1.5       | BSD License                                        |
| PyMuPDF                   | 1.24.7      | GNU AFFERO GPL 3.0                                 |
| PyMuPDFb                  | 1.24.6      | GNU AFFERO GPL 3.0                                 |
| PyYAML                    | 6.0.1       | MIT License                                        |
| Pygments                  | 2.18.0      | BSD License                                        |
| SQLAlchemy                | 2.0.31      | MIT License                                        |
| aiohttp                   | 3.9.5       | Apache Software License                            |
| aiosignal                 | 1.3.1       | Apache Software License                            |
| annotated-types           | 0.7.0       | MIT License                                        |
| anyio                     | 4.4.0       | MIT License                                        |
| asttokens                 | 2.4.1       | Apache 2.0                                         |
| async-timeout             | 4.0.3       | Apache Software License                            |
| attrs                     | 23.2.0      | MIT License                                        |
| backcall                  | 0.2.0       | BSD License                                        |
| beautifulsoup4            | 4.12.3      | MIT License                                        |
| bleach                    | 6.1.0       | Apache Software License                            |
| certifi                   | 2024.7.4    | Mozilla Public License 2.0 (MPL 2.0)               |
| charset-normalizer        | 3.3.2       | MIT License                                        |
| dataclasses-json          | 0.6.7       | MIT License                                        |
| datasets                  | 2.20.0      | Apache Software License                            |
| decorator                 | 5.1.1       | BSD License                                        |
| defusedxml                | 0.7.1       | Python Software Foundation License                 |
| dill                      | 0.3.8       | BSD License                                        |
| distro                    | 1.9.0       | Apache Software License                            |
| docopt                    | 0.6.2       | MIT License                                        |
| environs                  | 9.5.0       | MIT License                                        |
| et-xmlfile                | 1.1.0       | MIT License                                        |
| exceptiongroup            | 1.2.2       | MIT License                                        |
| executing                 | 2.0.1       | MIT License                                        |
| fastjsonschema            | 2.20.0      | BSD License                                        |
| filelock                  | 3.15.4      | The Unlicense (Unlicense)                          |
| frozenlist                | 1.4.1       | Apache Software License                            |
| fsspec                    | 2024.5.0    | BSD License                                        |
| greenlet                  | 3.0.3       | MIT License                                        |
| grpcio                    | 1.63.0      | Apache Software License                            |
| h11                       | 0.14.0      | MIT License                                        |
| httpcore                  | 1.0.5       | BSD License                                        |
| httpx                     | 0.27.0      | BSD License                                        |
| huggingface-hub           | 0.23.4      | Apache Software License                            |
| idna                      | 3.7         | BSD License                                        |
| ipython                   | 8.12.3      | BSD License                                        |
| jedi                      | 0.19.1      | MIT License                                        |
| joblib                    | 1.4.2       | BSD License                                        |
| jsonpatch                 | 1.33        | BSD License                                        |
| jsonpointer               | 3.0.0       | BSD License                                        |
| jsonschema                | 4.23.0      | MIT License                                        |
| jsonschema-specifications | 2023.12.1   | MIT License                                        |
| jupyter_client            | 8.6.2       | BSD License                                        |
| jupyter_core              | 5.7.2       | BSD License                                        |
| jupyterlab_pygments       | 0.3.0       | BSD License                                        |
| langchain                 | 0.2.7       | MIT License                                        |
| langchain-community       | 0.2.7       | MIT License                                        |
| langchain-core            | 0.2.17      | MIT License                                        |
| langchain-text-splitters  | 0.2.2       | MIT License                                        |
| langsmith                 | 0.1.83      | MIT License                                        |
| lxml                      | 5.2.2       | BSD License                                        |
| marshmallow               | 3.21.3      | MIT License                                        |
| matplotlib-inline         | 0.1.7       | BSD License                                        |
| milvus-lite               | 2.4.8       | UNKNOWN                                            |
| mistune                   | 3.0.2       | BSD License                                        |
| mpmath                    | 1.3.0       | BSD License                                        |
| multidict                 | 6.0.5       | Apache Software License                            |
| multiprocess              | 0.70.16     | BSD License                                        |
| mypy-extensions           | 1.0.0       | MIT License                                        |
| nbclient                  | 0.10.0      | BSD License                                        |
| nbconvert                 | 7.16.4      | BSD License                                        |
| nbformat                  | 5.10.4      | BSD License                                        |
| networkx                  | 3.3         | BSD License                                        |
| numpy                     | 1.26.4      | BSD License                                        |
| nvidia-cublas-cu12        | 12.1.3.1    | Other/Proprietary License                          |
| nvidia-cuda-cupti-cu12    | 12.1.105    | Other/Proprietary License                          |
| nvidia-cuda-nvrtc-cu12    | 12.1.105    | Other/Proprietary License                          |
| nvidia-cuda-runtime-cu12  | 12.1.105    | Other/Proprietary License                          |
| nvidia-cudnn-cu12         | 8.9.2.26    | Other/Proprietary License                          |
| nvidia-cufft-cu12         | 11.0.2.54   | Other/Proprietary License                          |
| nvidia-curand-cu12        | 10.3.2.106  | Other/Proprietary License                          |
| nvidia-cusolver-cu12      | 11.4.5.107  | Other/Proprietary License                          |
| nvidia-cusparse-cu12      | 12.1.0.106  | Other/Proprietary License                          |
| nvidia-nccl-cu12          | 2.20.5      | Other/Proprietary License                          |
| nvidia-nvjitlink-cu12     | 12.5.82     | Other/Proprietary License                          |
| nvidia-nvtx-cu12          | 12.1.105    | Other/Proprietary License                          |
| openai                    | 1.35.13     | Apache Software License                            |
| openpyxl                  | 3.1.5       | MIT License                                        |
| orjson                    | 3.10.6      | Apache Software License; MIT License               |
| packaging                 | 24.1        | Apache Software License; BSD License               |
| pandas                    | 2.2.2       | BSD License                                        |
| pandocfilters             | 1.5.1       | BSD License                                        |
| parso                     | 0.8.4       | MIT License                                        |
| pexpect                   | 4.9.0       | ISC License (ISCL)                                 |
| pickleshare               | 0.7.5       | MIT License                                        |
| pillow                    | 10.4.0      | Historical Permission Notice and Disclaimer (HPND) |
| pipreqs                   | 0.5.0       | Apache Software License                            |
| platformdirs              | 4.2.2       | MIT License                                        |
| prompt_toolkit            | 3.0.47      | BSD License                                        |
| protobuf                  | 5.27.2      | 3-Clause BSD License                               |
| ptyprocess                | 0.7.0       | ISC License (ISCL)                                 |
| pure-eval                 | 0.2.2       | MIT License                                        |
| pyarrow                   | 16.1.0      | Apache Software License                            |
| pyarrow-hotfix            | 0.6         | Apache License, Version 2.0                        |
| pydantic                  | 2.8.2       | MIT License                                        |
| pydantic_core             | 2.20.1      | MIT License                                        |
| pymilvus                  | 2.4.4       | Apache Software License                            |
| python-dateutil           | 2.9.0.post0 | Apache Software License; BSD License               |
| python-docx               | 1.1.2       | MIT License                                        |
| python-dotenv             | 1.0.1       | BSD License                                        |
| pytz                      | 2024.1      | MIT License                                        |
| pyzmq                     | 26.0.3      | BSD License                                        |
| referencing               | 0.35.1      | MIT License                                        |
| regex                     | 2024.5.15   | Apache Software License                            |
| requests                  | 2.32.3      | Apache Software License                            |
| rpds-py                   | 0.19.0      | MIT License                                        |
| safetensors               | 0.4.3       | Apache Software License                            |
| scikit-learn              | 1.5.1       | BSD License                                        |
| scipy                     | 1.14.0      | BSD License                                        |
| sentence-transformers     | 3.0.1       | Apache Software License                            |
| sentencepiece             | 0.2.0       | Apache Software License                            |
| six                       | 1.16.0      | MIT License                                        |
| sniffio                   | 1.3.1       | Apache Software License; MIT License               |
| soupsieve                 | 2.5         | MIT License                                        |
| stack-data                | 0.6.3       | MIT License                                        |
| sympy                     | 1.12.1      | BSD License                                        |
| tenacity                  | 8.5.0       | Apache Software License                            |
| threadpoolctl             | 3.5.0       | BSD License                                        |
| tinycss2                  | 1.3.0       | BSD License                                        |
| tokenizers                | 0.19.1      | Apache Software License                            |
| torch                     | 2.3.1       | BSD License                                        |
| tornado                   | 6.4.1       | Apache Software License                            |
| tqdm                      | 4.66.4      | MIT License; Mozilla Public License 2.0 (MPL 2.0)  |
| traitlets                 | 5.14.3      | BSD License                                        |
| transformers              | 4.42.3      | Apache Software License                            |
| triton                    | 2.3.1       | MIT License                                        |
| typing-inspect            | 0.9.0       | MIT License                                        |
| typing_extensions         | 4.12.2      | Python Software Foundation License                 |
| tzdata                    | 2024.1      | Apache Software License                            |
| ujson                     | 5.10.0      | BSD License                                        |
| urllib3                   | 2.2.2       | MIT License                                        |
| webencodings              | 0.5.1       | BSD License                                        |
| xxhash                    | 3.4.1       | BSD License                                        |
| yarg                      | 0.1.9       | MIT License                                        |
| yarl                      | 1.9.4       | Apache Software License                            |


See the [LICENSES](LICENSES) directory for details on third-party licenses.
