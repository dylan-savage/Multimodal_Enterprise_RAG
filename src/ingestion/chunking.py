from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer
from typing import Tuple, List
import numpy as np
import re
from .document_chunk import DocumentChunk

# Initialize models
CHUNKER = SemanticChunker(
    embedding_model="minishlab/potion-base-8M",
    threshold=0.7,
    chunk_size=1024,
    min_sentences=1
)

EMBEDDING_MODEL = SentenceTransformer("minishlab/potion-base-8M", device="cpu")

def clean_text(text: str) -> str:
    """Remove metadata and clean up text before chunking."""

    text = re.sub(r'\d{1,2}:\d{2} [AP]M [A-Z]{3,4}, [A-Za-z]+ \d{1,2}, \d{4}', '', text)
    
    text = re.sub(r'.*\.(jpg|jpeg|png|gif).*', '', text)
    
    text = re.sub(r'\d+ minute read', '', text)
    
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    
    return text

def semantic_chunk_and_embed(
    text: str,
    source_file: str,
    metadata: dict = None
) -> Tuple[List[DocumentChunk], np.ndarray]:
    """Chunk text semantically and generate embeddings for each chunk."""
    text = clean_text(text)
    
    if not text.strip():
        return [], np.array([])
    
    # Get chunks
    semantic_chunks = CHUNKER.chunk(text)
    
    # Extract text from SemanticChunk objects
    chunks = [chunk.text for chunk in semantic_chunks]
    
    # Generate embeddings
    inputs = [f"passage: {chunk}" for chunk in chunks]
    embeddings = EMBEDDING_MODEL.encode(inputs, convert_to_numpy=True)
    
    # Create document chunks
    doc_chunks = [
        DocumentChunk(
            content=chunk,
            source_file=source_file,
            chunk_index=i,
            metadata={
                **(metadata or {}),
                "length": len(chunk)
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    return doc_chunks, embeddings 