import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.text_ingestor import chunk_and_embed
from storage.keyword_index import KeywordIndex

chunks, _ = chunk_and_embed("data/news_article.txt")

index = KeywordIndex()
index.add_chunks(chunks)

