# Multimodal Enterprise RAG

A hybrid Retrieval-Augmented Generation (RAG) system designed to support enterprise-level question answering across multiple modalities — including text, images, and audio.

---

## Setup

### 1. Environment Setup

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Required System Tools

```bash
brew install tesseract  # for OCR
brew install ffmpeg     # for audio/video processing
```

### 3. Neo4j Setup

- Download and start a Neo4j instance locally
- Set these environment variables 
  ```env
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password
  ```

### 4. Qdrant Setup (Vector Database)

- Start Qdrant using Docker:
  ```bash
  # Start Qdrant container
  docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
  ```

- The container will be accessible at:
  - HTTP API: http://localhost:6333
  - gRPC: localhost:6334

- To stop the container:
  ```bash
  docker stop <container_name>
  ```

- To start it again:
  ```bash
  docker start <container_name>
  ```

### 5. Together.ai LLM Setup

- Add your API key in `.env`:

  ```env
  TOGETHER_API_KEY=your_key_here
  ```

---

## Pushing Data
To ingest data into the system, place your files into the `data/` directory and run the hybrid storage script:

```bash
# Push data to all storage systems (Qdrant, Neo4j, and Whoosh)
PYTHONPATH=src python src/scripts/push_to_hybrid_storage.py <file_path>
```

The script supports multiple file types:
- Text files (.txt, .md, .pdf, .html)
- Image files (.jpg, .jpeg, .png, .gif, .bmp)
- Audio files (.mp3, .wav, .m4a, .ogg)

---

## Running Tests

### Run all tests

```bash
PYTHONPATH=src python -m pytest tests/
```

### Run specific test ex:

```bash
PYTHONPATH=src python -m pytest tests/test_text_ingestor.py
```

---

## Running the Pipeline

```bash
PYTHONPATH=src python src/crew_pipeline/main_pipeline.py
```

---
## What works
- multimodal ingestion
- Hybrid Storage and retrieval
- Entity and Relationship extraction
- CrewAI agent pipeline

## To Improve

- **Answer hallucinations in non lookup queries**
- **eval log output for each query**
- **Stronger entity relationships**
- **Better Documentation**
- **Query Type handling with specific retrieval techniques**

## Notes

- Tested on Python 3.11.7
- Used:
  - [CrewAI](https://github.com/joaomdmoura/crewAI)
  - [Qdrant](https://qdrant.tech/)
  - [Neo4j](https://neo4j.com/)
  - [Together.ai](https://www.together.ai/)

---
