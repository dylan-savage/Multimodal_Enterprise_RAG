
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.text_ingestor import chunk_and_embed
from storage.graph_storage import GraphStorage
from processing.data_extraction import extract_graph_data_from_chunk

file_path = "data/news_article.txt"

chunks, _ = chunk_and_embed(file_path) 

graph = GraphStorage()

for chunk in chunks:
    extracted = extract_graph_data_from_chunk(chunk)
    graph.store_extracted_data(extracted, source_file=chunk.source_file)

graph.close()
print(f"Pushed {len(chunks)} chunks to Neo4j from {file_path}")