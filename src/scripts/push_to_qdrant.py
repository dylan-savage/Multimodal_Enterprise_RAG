import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.text_ingestor import chunk_and_embed
from storage.qdrant_client import QdrantStorage

file_path = "data/news_article.txt"

chunks, embeddings = chunk_and_embed(file_path)

qdrant = QdrantStorage()
qdrant.store_chunks(chunks, embeddings)
