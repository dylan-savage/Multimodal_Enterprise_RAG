import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.text_ingestor import chunk_and_embed
from ingestion.image_ingestor import ingest_image
from ingestion.audio_ingestor import ingest_audio
from storage.qdrant_client import QdrantStorage
from storage.graph_storage import GraphStorage
from storage.keyword_index import KeywordIndex
from processing.data_extraction import extract_graph_data_from_chunk

def get_file_type(file_path: str) -> str:
    """Determine the type of file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.txt', '.md', '.pdf', '.html']:
        return 'text'
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        return 'image'
    elif ext in ['.mp3', '.wav', '.m4a', '.ogg']:
        return 'audio'
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def process_file(file_path: str):
    """Process a file and return chunks and embeddings."""
    file_type = get_file_type(file_path)
    
    if file_type == 'text':
        return chunk_and_embed(file_path)
    elif file_type == 'image':
        chunks, embeddings = ingest_image(file_path)
        return chunks, embeddings
    elif file_type == 'audio':
        chunks, embeddings = ingest_audio(file_path)
        return chunks, embeddings

def push_to_all_storage(file_path: str):
    """Push file content to all storage systems."""
    logging.info(f"Processing file: {file_path}")
    
    # Process the file
    chunks, embeddings = process_file(file_path)
    if not chunks:
        logging.warning(f"No content extracted from {file_path}")
        return
    
    # Initialize storage systems
    qdrant = QdrantStorage()
    graph = GraphStorage()
    keyword_index = KeywordIndex()
   
    # Push to Qdrant (vector storage)
    if embeddings.any():
        qdrant.store_chunks(chunks, embeddings)
    
    # Push to Neo4j (graph storage)
    for chunk in chunks:
        extracted = extract_graph_data_from_chunk(chunk)
        graph.store_extracted_data(extracted, source_file=chunk.source_file)
    
    # Push to Whoosh (keyword storage)
    logging.info(f"Adding {len(chunks)} chunks to Whoosh index.")
    keyword_index.add_chunks(chunks)
    
    # Close Neo4j connection
    graph.close()
    
    logging.info(f"Successfully pushed {len(chunks)} chunks from {file_path} to all storage systems")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python push_to_hybrid_storage.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    
    push_to_all_storage(file_path)
