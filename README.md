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

## Quick Start

The fastest way to get started:

~~~sh
# 1. Install dependencies
uv sync

# 2. Start ChromaDB (optional - can use embedded mode)
./scripts/run-chromadb.sh

# 3. Install and run Ollama (in a separate terminal)
# Download from https://ollama.ai
ollama pull gemma2:9b
ollama pull nomic-embed-text:latest

# 4. Ingest your support case files
uv run python src/cli.py local-ingest \
  -d ./case_files \
  -m nomic-embed-text:latest \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -ek "not-needed"

# 5. Start the RAG API
uv run python src/cli.py rag-api \
  -m gemma2:9b \
  -em nomic-embed-text:latest \
  -llm-api http://127.0.0.1:11434/v1 \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -lk "not-needed" \
  -ek "not-needed"

# 6. Start the chatbot (in another terminal)
uv run python src/cli.py chatbot
~~~

## Prerequisites

Before running the tool, you need to set up the following services:

### 1. ChromaDB (Vector Database)

ChromaDB is required for storing document embeddings. You can run it using the provided script:

~~~sh
# Using the provided script:
./scripts/run-chromadb.sh

# Or manually with Docker/Podman:
mkdir -p /var/tmp/chroma-data
podman run -d --rm --name chromadb \
  -v /var/tmp/chroma-data:/data:rw,z \
  -p 8000:8000 \
  docker.io/chromadb/chroma:latest
~~~

ChromaDB will be available at `http://localhost:8000`.

> **NOTE**: You can also use ChromaDB in embedded mode (default) by not specifying `--db-endpoint` in the CLI commands. This creates a local `./chromadb` directory.

### 2. LLM API Endpoints

You have two options for LLM services:

**Option A: Ollama (Local)**
- Install Ollama from [ollama.ai](https://ollama.ai)
- Pull the models you want to use:
  ~~~sh
  ollama pull gemma2:9b
  ollama pull nomic-embed-text:latest
  ~~~
- Ollama API will be available at `http://localhost:11434`

**Option B: Google Gemini (Cloud)**
- Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Set it in `.env` file or pass via CLI flag

## How to run the tool

### 1. Install requirements

~~~sh
uv sync
~~~

### 2. Run data ingestion

Ingest support case files from a local directory:

> **NOTE**: By default, the ingestion will only ingest data not existing in the Vector DB. Use `-i` parameter to initialize the db from scratch.

~~~sh
# Using embedded ChromaDB (creates ./chromadb directory):
uv run python src/cli.py local-ingest \
  -d ./case_files \
  -m nomic-embed-text:latest \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -ek "not-needed"

# Using ChromaDB container (if you started it with the script):
uv run python src/cli.py local-ingest \
  -d ./case_files \
  -m nomic-embed-text:latest \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -ek "not-needed" \
  --db-endpoint http://127.0.0.1:8000
~~~

### 3. Run the RAG API

Start the RAG API server that handles query processing:

~~~sh
# Example with Ollama and embedded ChromaDB:
uv run python src/cli.py rag-api \
  -m gemma2:9b \
  -em nomic-embed-text:latest \
  -llm-api http://127.0.0.1:11434/v1 \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -lk "not-needed" \
  -ek "not-needed" \
  -p 8080

# With ChromaDB container:
uv run python src/cli.py rag-api \
  -m gemma2:9b \
  -em nomic-embed-text:latest \
  -llm-api http://127.0.0.1:11434/v1 \
  -em-api http://127.0.0.1:11434/api/embeddings \
  -lk "not-needed" \
  -ek "not-needed" \
  --db-endpoint http://127.0.0.1:8000 \
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

### 6. Run the A2A Agent Server (Advanced)

Expose the agent via the Agent-to-Agent (A2A) protocol for programmatic access:

> **NOTE**: Requires RAG API to be running (see step 3)

**What is A2A?**
The Agent-to-Agent (A2A) protocol allows agents to discover and communicate with each other. This is useful for building multi-agent systems where agents can delegate tasks to specialized agents.

**Starting the A2A Agent Server:**

~~~sh
# Using Google Gemini (recommended):
uv run python src/cli.py a2a-agent \
  -m gemini-2.0-flash \
  --rag-api-endpoint http://127.0.0.1:8080/answer \
  -p 8001

# Using Ollama:
uv run python src/cli.py a2a-agent \
  -m gemma2:9b \
  -llm-api http://127.0.0.1:11434/v1 \
  -lk "not-needed" \
  --rag-api-endpoint http://127.0.0.1:8080/answer \
  -p 8001
~~~

**Accessing the Agent Card:**

Once the server is running, you can view the agent's capabilities at:
```
http://localhost:8001/.well-known/agent-card.json
```

This endpoint exposes auto-generated metadata about the agent, including:
- Agent name and description
- Available capabilities (automatically extracted from agent tools)
- Input/output modes
- Protocol version information

**Testing the A2A Agent:**

**Using the Test Script:**

Run the included test script to quickly verify your A2A agent is working:

~~~sh
uv run python test_a2a_agent.py
~~~

This script will:
- Connect to the A2A agent at `http://localhost:8001`
- Fetch and display the agent card
- Send a test conversation to verify the agent's functionality

**Programmatic Integration:**

Other agents or applications can discover and interact with this agent programmatically using the A2A protocol. The agent card provides all the information needed for automatic integration.
