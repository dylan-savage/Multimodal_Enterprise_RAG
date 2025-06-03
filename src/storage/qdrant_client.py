from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from typing import List
import uuid
from qdrant_client.http.models import SearchParams
from ingestion.document_chunk import DocumentChunk
from sentence_transformers import SentenceTransformer


class QdrantStorage:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "multimodal_chunks", vector_size: int = 256):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.model = SentenceTransformer("minishlab/potion-base-8M", device="cpu")

        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def store_chunks(self, chunks: List[DocumentChunk], embeddings):
        points = []

        for chunk, vector in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "content": chunk.content,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    **(chunk.metadata or {})
                }
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Uploaded {len(points)} chunks to Qdrant.")

    def search_chunks(self, query: str, top_k: int = 5) -> List[dict]:

        try:
            query_vector = self.model.encode(f"passage: {query}").tolist()

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=512)
            )

            return [
                {
                    "text": hit.payload.get("content", ""),
                    "score": hit.score,
                    "source_file": hit.payload.get("source_file"),
                    "chunk_index": hit.payload.get("chunk_index"),
                }
                for hit in results
            ]
        except Exception as e:
            return [{"error": str(e)}]
