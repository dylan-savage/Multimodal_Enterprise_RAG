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

### 4. Together.ai LLM Setup

- Add your API key in `.env`:

  ```env
  TOGETHER_API_KEY=your_key_here
  ```

---

## Pushing Data
To ingest data into the system, place your files into the `data/` directory and run the relevant ingestion scripts after updating the file path:

```bash
# Push structured chunks to Qdrant (semantic vector search)
PYTHONPATH=src python src/scripts/push_to_qdrant.py

# Push entity/relationship triples to Neo4j (graph traversal)
PYTHONPATH=src python src/scripts/push_to_neo.py

# Index documents for keyword search using Whoosh
PYTHONPATH=src python src/scripts/test_whoosh.py
```

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

## Remaining Work

- **Retrieval within Crew pipeline inconsistent**
  While storage methods perform well in test scripts, query formatting inside CrewAI tools sometimes break retrieval.

- **Answer hallucination due to weak or missing context**  
  If retrieval fails or returns irrelevant data, the Generator may hallucinate answers. This is especially apparent when the Reranker has little to work with.

- **No UI or interactive demo**  
  The current system runs entirely via CLI. 

- **Graph depth and schema coverage**  
  Neo4j holds only shallow relationships (e.g., 1-hop facts like `NFL —[has]→ committee`). Expanding entity-linking and enriching with deeper or cross-modal nodes would make for better reasoning.

- **Evaluation framework is identified but not applied**  
  A structured evaluation plan was outlined, targeting key metrics:
  - Faithfulness
  - Retrieval Relevance
  - Answer Correctness
  Query types considered included look-up, reasoning, and cross-modal (MVP), with stretch goals summarization and sentiment analysis. 
  Formal test cases and metric scoring with deepeval have not yet been implemented. This means that there is also no evaluation output after each query.

- **Smaller Items**
  - Clarifier loopback
  - LLaVA for image ingestion
  - Data Extraction from graph
    - graph validation logic — Prevent malformed data or duplicates
    - Filtering by predicate
    - More test coverage for graph pipeline
  - Format pushing techniques for large amounts of data to all three storages
  - Modal-Specific Queries
  - Crew retry logic and graceful failure handling


## Notes

- Tested on Python 3.11.7
- Used:
  - [CrewAI](https://github.com/joaomdmoura/crewAI)
  - [Qdrant](https://qdrant.tech/)
  - [Neo4j](https://neo4j.com/)
  - [Together.ai](https://www.together.ai/)

---
