import os
from chonkie.chunker.semantic import SemanticChunker
from sentence_transformers import SentenceTransformer
from .loaders import load_text_from_file
from .document_chunk import DocumentChunk

def chunk_and_embed(file_path: str):
    print(f"Processing: {file_path}")
    text = load_text_from_file(file_path)

    chunker = SemanticChunker(return_type="texts")
    chunks = chunker.chunk(text)

    model = SentenceTransformer("minishlab/potion-base-8M", device="cpu")
    inputs = [f"passage: {chunk}" for chunk in chunks]
    embeddings = model.encode(inputs, convert_to_numpy=True)

    doc_chunks = [
        DocumentChunk(
            content=chunk,
            source_file=os.path.basename(file_path),
            chunk_index=i,
            metadata={"modality": "text","length": len(chunk)}
        )
        for i, chunk in enumerate(chunks)
    ]

    return doc_chunks, embeddings
