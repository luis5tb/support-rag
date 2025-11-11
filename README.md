# Support Case RAG Tool

This is a PoC that aims to provide a RAG tool that can answer user queries by using previous support case data to back its answers.

## Architecture

This project uses a hybrid approach combining two powerful frameworks:

- **LlamaIndex**: Handles RAG functionality including document ingestion, embeddings, vector storage (ChromaDB), hybrid retrieval (BM25 + semantic search), and query engines
- **Google ADK (Agent Development Kit)**: Powers the conversational agent that gathers user information and orchestrates queries to the RAG system

This architecture provides the best of both worlds: LlamaIndex's mature RAG capabilities with ADK's advanced agent orchestration.

### LLM Options for Agent

The agent supports two deployment options:

1. **Google Gemini** (recommended): Use Google's Gemini models via API
   - Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
   - Set `GEMINI_API_KEY` in `.env` file or pass via `-lk` flag

2. **Ollama/OpenAI-compatible**: Use local Ollama or any OpenAI-compatible endpoint
   - Requires `-llm-api` flag to specify the endpoint
   - Can use any model available on your local Ollama instance

## Support Case Files Structure

The support case files ingested by the tool are expected to be formatted in Markdown and to have the following structure:

~~~code
# <case_number> - <case_title>
## Summary
<case_summary>
## Description
<case_description>
## Comments
### Comment 1
<comment_1>
### Comment N
<comment_n>
~~~

The files are expected to have the following naming:

* case_<case_number>.md

## How to run the tool

### 1. Install requirements

~~~sh
uv sync
~~~

### 2. Run data ingestion

Ingest support case files from a local directory:

> **NOTE**: By default, the ingestion will only ingest data not existing in the Vector DB. Use `-i` parameter to initialize the db from scratch.

~~~sh
uv run python src/cli.py local-ingest \
  -d <source_dir> \
  -m <embeddings_model> \
  -em-api <embeddings_api_endpoint> \
  -ek <embedding_api_key>

# Example with Ollama:
uv run python src/cli.py local-ingest \
  -d ./case_files \
  -m nomic-embed-text:latest \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -ek "not-needed"
~~~

### 3. Run the RAG API

Start the RAG API server that handles query processing:

~~~sh
uv run python src/cli.py rag-api \
  -m <llm_model> \
  -em <embeddings_model> \
  -llm-api <llm_api_endpoint> \
  -em-api <embeddings_api_endpoint> \
  -lk <llm_api_key> \
  -ek <embedding_api_key> \
  -p <port>

# Example with Ollama:
uv run python src/cli.py rag-api \
  -m gemma2:9b \
  -em nomic-embed-text:latest \
  -llm-api http://127.0.0.1:11434/v1 \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -lk "not-needed" \
  -ek "not-needed" \
  -p 8080
~~~

### 4. Run the Chatbot WebUI

Simple chatbot interface that directly queries the RAG API:

> **NOTE**: Requires RAG API to be running (see step 3)

~~~sh
uv run python src/cli.py chatbot \
  --rag-api-endpoint http://127.0.0.1:8080/answer \
  -p 8181
~~~

### 5. Run the Agent WebUI (Recommended)

Advanced agent interface powered by Google ADK that gathers user information before querying:

> **NOTE**: Requires RAG API to be running (see step 3)

**Option A: Using Google Gemini (recommended)**

~~~sh
# If GEMINI_API_KEY is set in .env file:
uv run python src/cli.py agent \
  -m gemini-2.0-flash \
  --rag-api-endpoint http://127.0.0.1:8080/answer \
  -p 8282

# Or pass the API key directly:
uv run python src/cli.py agent \
  -m gemini-2.0-flash \
  -lk "your-gemini-api-key" \
  --rag-api-endpoint http://127.0.0.1:8080/answer \
  -p 8282
~~~

Get a Gemini API key: [Google AI Studio](https://aistudio.google.com/apikey)

**Option B: Using Ollama (local deployment)**

~~~sh
uv run python src/cli.py agent \
  -m gemma2:9b \
  -llm-api http://127.0.0.1:11434/v1 \
  -lk "not-needed" \
  --rag-api-endpoint http://127.0.0.1:8080/answer \
  -p 8282
~~~

The agent interface provides a conversational experience that:
- Gathers required information (product version, error message, etc.)
- Validates and stores user input
- Automatically queries the RAG system when all information is collected
