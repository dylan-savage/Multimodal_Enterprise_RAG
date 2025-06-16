from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser, QueryParser, OrGroup
from whoosh import scoring
import os
from typing import List
import logging

from ingestion.document_chunk import DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeywordIndex:
    def __init__(self, index_dir="whoosh_index"):
        self.index_dir = index_dir
        logging.info(f"Using Whoosh index directory: {os.path.abspath(index_dir)}")
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
            schema = Schema(
                chunk_id=ID(stored=True, unique=True),
                content=TEXT(stored=True),
                source_file=ID(stored=True),
                chunk_index=NUMERIC(stored=True),
                modality=ID(stored=True)
            )
            self.index = create_in(index_dir, schema)
        else:
            self.index = open_dir(index_dir)

    def add_chunks(self, chunks: List[DocumentChunk]):
        writer = self.index.writer()
        for chunk in chunks:
            writer.add_document(
                chunk_id=f"{chunk.source_file}-{chunk.chunk_index}",
                content=chunk.content,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                modality=chunk.metadata.get("modality", "unknown") if chunk.metadata else "unknown"
            )
        logging.info("Committing changes to Whoosh index.")
        writer.commit()
        logging.info(f"Indexed {len(chunks)} chunks in Whoosh.")


    def search(self, query: str, top_k=5):
        ix = open_dir(self.index_dir)
        with ix.searcher(weighting=scoring.BM25F()) as searcher:
            parser = QueryParser("content", ix.schema, group=OrGroup)

            # Simplify query: remove words like 'definition', 'explain', etc.
            stop_words = {"definition", "explain", "describe", "what", "is", "the"}
            terms = [term for term in query.lower().split() if term not in stop_words]
            simplified_query = " OR ".join(terms)  # e.g., "tush OR push"

            parsed_query = parser.parse(simplified_query)
            results = searcher.search(parsed_query, limit=top_k)
            return [dict(hit) for hit in results]


    def clear_database(self) -> None:
        """Delete all documents from the index."""
        writer = self.index.writer()
        # Create a query that matches all documents
        parser = QueryParser("content", self.index.schema, group=OrGroup)
        query = parser.parse("*")  # Match all documents
        writer.delete_by_query(query)
        writer.commit()
        print("Cleared Whoosh database.")
