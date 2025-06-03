from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
from whoosh import scoring
import os
from typing import List

from ingestion.document_chunk import DocumentChunk

class KeywordIndex:
    def __init__(self, index_dir="whoosh_index"):
        self.index_dir = index_dir
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

        self.writer = self.index.writer()


    def add_chunks(self, chunks: List[DocumentChunk]):
        for chunk in chunks:
            self.writer.add_document(
                chunk_id=f"{chunk.source_file}-{chunk.chunk_index}",
                content=chunk.content,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                modality=chunk.metadata.get("modality", "unknown") if chunk.metadata else "unknown"
            )
        self.writer.commit()
        print(f"Indexed {len(chunks)} chunks in Whoosh.")

    def search(self, query: str, top_k=5):
        ix = open_dir(self.index_dir)
        with ix.searcher(weighting=scoring.BM25F()) as searcher:
            parser = MultifieldParser(["content", "source_file", "modality"], schema=ix.schema)
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=top_k)
            return [dict(hit) for hit in results]
